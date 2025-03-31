from typing import Tuple

import torch
from jaxtyping import Float, Bool, Int
from torch.nn.modules import activation
from pbd_torch.model import State
from pbd_torch.transform import rotate_vectors_batch
from pbd_torch.transform import transform_multiply_batch
from pbd_torch.transform import transform_points_batch
from pbd_torch.ncp import ScaledFisherBurmeister
from pbd_torch.model import Model
from pbd_torch.ncp import FisherBurmeister


def skew_symmetric_matrix_batch(
    vectors: Float[torch.Tensor, "N 3 1"]
) -> Float[torch.Tensor, "N 3 3"]:
    """
    Compute the skew-symmetric matrix of a batch of vectors.

    Args:
        vectors: Tensor of shape [N, 3, 1].

    Returns:
        Tensor of shape [N, 3, 3] with skew-symmetric matrices.
    """

    N = vectors.shape[0]
    skew = torch.zeros(N, 3, 3, device=vectors.device, dtype=vectors.dtype)
    skew[:, 0, 1] = -vectors[:, 2].squeeze(-1)
    skew[:, 0, 2] = vectors[:, 1].squeeze(-1)
    skew[:, 1, 0] = vectors[:, 2].squeeze(-1)
    skew[:, 1, 2] = -vectors[:, 0].squeeze(-1)
    skew[:, 2, 0] = -vectors[:, 1].squeeze(-1)
    skew[:, 2, 1] = vectors[:, 0].squeeze(-1)
    return skew

class DynamicsConstraint:
    def __init__(self, mass_matrix: torch.Tensor,
                 g_accel: torch.Tensor,
                 device: torch.device):
        self.device = device
        self.mass_matrix = mass_matrix
        self.g_accel = g_accel.expand(mass_matrix.shape[0], 6, 1)

    def get_residuals(self,
        body_vel: Float[torch.Tensor, "body_count 6 1"],
        body_vel_prev: Float[torch.Tensor, "body_count 6 1"],
        lambda_n: Float[torch.Tensor, "body_count max_contacts 1"],
        lambda_t: Float[torch.Tensor, "body_count 2 * max_contacts 1"],
        body_f: Float[torch.Tensor, "body_count 6 1"],
        J_n: Float[torch.Tensor, "body_count max_contacts 6"],
        J_t: Float[torch.Tensor, "body_count 2 * max_contacts 6"],
        dt: float,
    ) -> Float[torch.Tensor, "body_count 6 1"]:
        return (
            torch.matmul(self.mass_matrix, (body_vel - body_vel_prev))  # M @ (v - v_prev)
            - torch.matmul(J_n.transpose(2, 1), lambda_n)  # J_n^T @ λ_n
            - torch.matmul(J_t.transpose(2, 1), lambda_t)  # J_t^T @ λ_t
            - body_f * dt  # f·dt
            - torch.matmul(self.mass_matrix, self.g_accel) * dt  # M @ g·dt
        )

    def get_derivatives(self,
         body_vel: Float[torch.Tensor, "body_count 6 1"],
         body_vel_prev: Float[torch.Tensor, "body_count 6 1"],
         lambda_n: Float[torch.Tensor, "body_count max_contacts 1"],
         lambda_t: Float[torch.Tensor, "body_count 2 * max_contacts 1"],
         body_f: Float[torch.Tensor, "body_count 6 1"],
         J_n: Float[torch.Tensor, "body_count max_contacts 6"],
         J_t: Float[torch.Tensor, "body_count 2 * max_contacts 6"],
         dt: float,
         ) -> Tuple[Float[torch.Tensor, "body_count 6 6"],
             Float[torch.Tensor, "body_count max_contacts 6"],
             Float[torch.Tensor, "body_count 2 * max_contacts 6"]]:
        # ∂res/∂body_vel
        dres_dbody_vel = self.mass_matrix # [B, 6, 6]

        # ∂res/∂lambda_n
        dres_dlambda_n = -J_n.transpose(1, 2) # [B, 6, C]

        # ∂res/∂lambda_t
        dres_dlambda_t = -J_t.transpose(1, 2) # [B, 6, 2C]

        return dres_dbody_vel, dres_dlambda_n, dres_dlambda_t


class ContactConstraint:
    def __init__(self,
        device: torch.device):
        self.device = device

        self.stabilization_factor = 0.2
        self.fb = ScaledFisherBurmeister(alpha=0.3, beta=0.3, epsilon=1e-12)

    def get_penetration_depths(self,
        body_trans: Float[torch.Tensor, "body_count 7 1"],
        contact_points: Float[torch.Tensor, "body_count max_contacts 3 1"],
        ground_points: Float[torch.Tensor, "body_count max_contacts 3 1"],
        contact_normals: Float[torch.Tensor, "body_count max_contacts 3 1"]
    ) -> Float[torch.Tensor, "body_count max_contacts 1"]:
        B = body_trans.shape[0]  # Body count
        C = contact_points.shape[1]  # Max contacts per body

        # Transform contact points from local to world frame
        expanded_body_transforms = body_trans.unsqueeze(1).expand(B, C, 7, 1)
        points_world = transform_points_batch(contact_points, expanded_body_transforms)  # [B, C, 3, 1]

        # Calculate penetration as the projection of (ground_point - contact_point) onto normal
        diff = ground_points - points_world  # [B, C, 3, 1]
        penetration = (diff * contact_normals).sum(dim=2)  # [B, C, 1]

        return penetration

    def compute_contact_jacobians(self,
        body_trans: Float[torch.Tensor, "body_count 7 1"],
        contact_points: Float[torch.Tensor, "body_count max_contacts 3 1"],
        contact_normals: Float[torch.Tensor, "body_count max_contacts 3 1"],
        contact_mask: Bool[torch.Tensor, "body_count max_contacts"],
    ) -> Float[torch.Tensor, "body_count max_contacts 6"]:
        """Compute the batched contact Jacobian for all bodies and their contacts.

        Args:
            body_q: Body transforms in quaternion format [pos, quat]
            contact_points: Contact points in local body frame
            contact_normals: Contact normals in world frame
            contact_mask: Boolean mask indicating active contacts

        Returns:
            Contact Jacobian for normal directions
        """
        B = body_trans.shape[0]  # Number of bodies
        C = contact_mask.shape[1]  # Max contacts per body

        # Initialize the Jacobian with zeros
        J = torch.zeros((B, C, 6), device=self.device)  # [B, C, 6]

        # Transform local contact points to world frame
        q = body_trans[:, 3:].unsqueeze(1).expand(B, C, 4, 1)  # [B, C, 4, 1]
        r = rotate_vectors_batch(contact_points, q)  # [B, C, 3, 1]

        # Compute r × n for rotational component of the Jacobian
        r_cross_n = torch.cross(r, contact_normals, dim=2)  # [B, C, 3, 1]

        J = torch.cat([r_cross_n, contact_normals], dim=2).squeeze(-1)  # [B, C, 6]
        J[~contact_mask] = 0

        return J

    def get_residuals(self,
        body_vel: Float[torch.Tensor, "body_count 6 1"],
        body_vel_prev: Float[torch.Tensor, "body_count 6 1"],
        lambda_n: Float[torch.Tensor, "body_count max_contacts 1"],
        J_n: Float[torch.Tensor, "body_count max_contacts 6"],
        penetration_depth: Float[torch.Tensor, "body_count max_contacts 1"],
        contact_mask: Bool[torch.Tensor, "body_count max_contacts"],
        restitution: Float[torch.Tensor, "body_count 1"],
        dt: float,
    ) -> Float[torch.Tensor, "body_count max_contacts 1"]:
        B = body_vel.shape[0] # Body count
        C = J_n.shape[1] # Maximum contact count per body

        active_mask = contact_mask.view(B, C, 1).float() # [B, C, 1]
        inactive_mask = 1 - active_mask # [B, C, 1]

        v_n = torch.matmul(J_n, body_vel) # [B, C, 6] @ [B, 6, 1] -> [B, C, 1]
        v_n_prev = torch.matmul(J_n, body_vel_prev) # [B, C, 6] @ [B, 6, 1] -> [B, C, 1]
        e = restitution.view(B, 1, 1).expand(B, C, 1) # [B, C, 1]

        b_err = -(self.stabilization_factor / dt) * penetration_depth  # [B, C, 1]
        b_rest = e * v_n_prev # [B, C, 6] @ [B, C, 1] -> [B, C, 1]

        res_act = self.fb.evaluate(lambda_n, v_n + b_err + b_rest) # [B, C, 1]
        res_inact = -lambda_n # [B, C, 1]

        res = active_mask * res_act + inactive_mask * res_inact # [B, C, 1]

        return res # [B, C, 1]

    def get_derivatives(self,
        body_vel: Float[torch.Tensor, "body_count 6 1"],
        body_vel_prev: Float[torch.Tensor, "body_count 6 1"],
        lambda_n: Float[torch.Tensor, "body_count max_contacts 1"],
        J_n: Float[torch.Tensor, "body_count max_contacts 6"],
        penetration_depth: Float[torch.Tensor, "body_count max_contacts 1"],
        contact_mask: Bool[torch.Tensor, "body_count max_contacts"],
        restitution: Float[torch.Tensor, "body_count 1"],
        dt: float,
    ) -> Tuple[Float[torch.Tensor, "body_count max_contacts 6"],
        Float[torch.Tensor, "body_count max_contacts max_contacts"]]:
        B = body_vel.shape[0] # Body count
        C = J_n.shape[1] # Maximum contact count per body

        active_mask = contact_mask.view(B, C, 1).float() # [B, C, 1]
        inactive_mask = 1 - active_mask # [B, C, 1]

        v_n = torch.matmul(J_n, body_vel) # [B, C, 6] @ [B, 6, 1] -> [B, C, 1]
        v_n_prev = torch.matmul(J_n, body_vel_prev) # [B, C, 6] @ [B, 6, 1] -> [B, C, 1]
        e = restitution.view(B, 1, 1).expand(B, C, 1) # [B, C, 1]

        b_err = -(self.stabilization_factor / dt) * penetration_depth  # [B, C, 1]
        b_rest = e * v_n_prev # [B, C, 6] @ [B, C, 1] -> [B, C, 1]

        da_act, db_act = self.fb.derivatives(lambda_n, v_n + b_err + b_rest) # [B, C, 1]
        da_n_inact = -torch.ones_like(lambda_n) # [B, C, 1]
        db_n_inact = torch.zeros_like(lambda_n) # [B, C, 1]

        da = da_act * active_mask + da_n_inact * inactive_mask # [B, C, 1]
        db = db_act * active_mask + db_n_inact * inactive_mask # [B, C, 1]

        # ∂res/∂body_vel
        dres_dbody_vel = db * J_n # [B, C, 1] * [B, C, 6] -> [B, C, 6]

        # ∂res/∂lambda_n
        dres_dlambda_n = torch.diag_embed(da.view(B, C)) # [B, C, C]

        return dres_dbody_vel, dres_dlambda_n


class FrictionConstraint:
    def __init__(self, device: torch.device):
        self.device = device
        self.eps = 1e-12

        self.fb = ScaledFisherBurmeister(alpha=0.3, beta=0.3, epsilon=self.eps)
        # self.fb = FisherBurmeister(epsilon=self.eps)

    @staticmethod
    def _compute_tangential_basis(
        contact_normals: Float[torch.Tensor, "body_count max_contacts 3 1"]
    ) -> Tuple[
        Float[torch.Tensor, "body_count max_contacts 3 1"],
        Float[torch.Tensor, "body_count max_contacts 3 1"]
    ]:
        """Compute orthogonal tangent vectors t1 and t2 for each contact normal.

        Args:
            contact_normals: Contact normals in world frame

        Returns:
            Tuple of tangent vectors (t1, t2) orthogonal to the contact normals
        """
        device = contact_normals.device
        B = contact_normals.shape[0] # Body count
        C = contact_normals.shape[1] # Contact count

        # Choose a reference vector not parallel to normal (default: [1, 0, 0])
        ref = torch.tensor([1.0, 0.0, 0.0], device=device)
        ref = ref.repeat(B, C, 1).unsqueeze(-1)  # [B, C, 3, 1]

        # For nearly parallel cases, switch reference to [0, 1, 0]
        mask = (torch.abs(torch.sum(contact_normals * ref, dim=2)) > 0.99).squeeze(-1)
        ref[mask] = torch.tensor([0.0, 1.0, 0.0], device=device).view(3, 1)

        # Compute first tangent using cross product
        t1 = torch.cross(contact_normals, ref, dim=2) # [B, C, 3, 1]
        t1 = t1 / (torch.norm(t1, dim=2, keepdim=True) + 1e-6)  # Normalize

        # Compute second tangent perpendicular to both normal and first tangent
        t2 = torch.cross(contact_normals, t1, dim=2) # [B, C, 3, 1]
        t2 = t2 / (torch.norm(t2, dim=2, keepdim=True) + 1e-6)  # Normalize

        return t1, t2

    def compute_tangential_jacobians(self,
        body_trans: Float[torch.Tensor, "body_count 7 1"],
        contact_points: Float[torch.Tensor, "body_count max_contacts 3 1"],
        contact_normals: Float[torch.Tensor, "body_count max_contacts 3 1"],
        contact_mask: Float[torch.Tensor, "body_count max_contacts"]
    ) -> Float[torch.Tensor, "body_count 2*max_contacts 6"]:
        """Compute the tangential Jacobian for each contact in both t1 and t2 directions.

        Args:
            body_q: Body transforms in quaternion format [pos, quat]
            contact_points: Contact points in local body frame
            contact_normals: Contact normals in world frame

        Returns:
            Tangential Jacobian for friction directions
        """
        B = body_trans.shape[0] # Number of bodies
        C = contact_points.shape[1] # Max number of contacts per body

        # Compute tangent basis vectors for each contact normal
        t1, t2 = self._compute_tangential_basis(contact_normals)  # [B, C, 3, 1]

        # Transform local contact points to world frame
        q = body_trans[:, 3:].unsqueeze(1).expand(B, C, 4, 1)  # [B, C, 4, 1]
        r = rotate_vectors_batch(contact_points, q)  # [B, C, 3, 1]

        # Compute Jacobians: J = [(r × t), t] for both tangent directions
        r_cross_t1 = torch.cross(r, t1, dim=2)  # [B, C, 3, 1]
        r_cross_t2 = torch.cross(r, t2, dim=2)  # [B, C, 3, 1]

        # Assemble Jacobians for both tangent directions
        J_t1 = torch.cat([-r_cross_t1, -t1], dim=2).squeeze(-1)  # [B, C, 6]
        J_t1[~contact_mask] = 0.0
        J_t2 = torch.cat([-r_cross_t2, -t2], dim=2).squeeze(-1)  # [B, C, 6]
        J_t2[~contact_mask] = 0.0

        # Stack into a single tensor with both directions
        J_t = torch.cat((J_t1, J_t2), dim=1)  # [B, 2*C, 6]

        return J_t

    def get_residuals(self,
        body_vel: Float[torch.Tensor, "body_count 6 1"],
        lambda_n: Float[torch.Tensor, "body_count max_contacts 1"],
        lambda_t: Float[torch.Tensor, "body_count 2*max_contacts 1"],
        gamma: Float[torch.Tensor, "body_count max_contacts 1"],
        J_t: Float[torch.Tensor, "body_count 2*max_contacts 6"],
        contact_mask: Bool[torch.Tensor, "body_count max_contacts"],
        friction_coeff: Float[torch.Tensor, "body_count 1"]
    ) -> Float[torch.Tensor, "body_count 3*max_contacts 1"]:
        B = body_vel.shape[0]
        C = lambda_n.shape[1]

        active_mask = contact_mask.view(B, C, 1).float() # [B, C, 1]
        inactive_mask = 1 - active_mask # [B, C, 1]

        # Compute tangential velocity
        v_t = torch.matmul(J_t, body_vel)  # [B, 2 * C, 1]
        v_t1 = v_t[:, :C, :]  # [B, C, 1]
        v_t2 = v_t[:, C:, :]  # [B, C, 1]

        # Expand the friction coefficient
        mu = friction_coeff.unsqueeze(1).expand(B, C, 1)  # [B, C, 1]

        # Split lambda_t into two parts
        lambda_t1 = lambda_t[:, :C, :]  # [B, C, 1]
        lambda_t2 = lambda_t[:, C:, :]  # [B, C, 1]

        # Compute the friction impulse norm
        lambda_t_norm = torch.sqrt(lambda_t1 ** 2 + lambda_t2 ** 2 + self.eps)  # [B, C, 1]

        # Compute residuals for friction directions
        res_fr1_act = v_t1 + gamma * lambda_t1 / (lambda_t_norm + self.eps)  # [B, C, 1]
        res_fr2_act = v_t2 + gamma * lambda_t2 / (lambda_t_norm + self.eps)  # [B, C, 1]

        res_fr1_inact = -lambda_t1  # [B, C, 1]
        res_fr2_inact = -lambda_t2  # [B, C, 1]

        res_fr1 = res_fr1_act * active_mask + res_fr1_inact * inactive_mask  # [B, C, 1]
        res_fr2 = res_fr2_act * active_mask + res_fr2_inact * inactive_mask  # [B, C, 1]

        res_fr = torch.cat([res_fr1, res_fr2], dim=1)  # [B, 2*C, 1]

        res_frc_act = self.fb.evaluate(gamma, mu * lambda_n - lambda_t_norm) # [B, C, 1]
        res_frc_inact = - gamma # [B, C, 1]
        res_frc = res_frc_act * active_mask + res_frc_inact * inactive_mask # [B, C, 1]

        res = torch.cat([res_fr, res_frc], dim=1) # [B, 3 * C, 1]

        return res

    def get_derivatives(self,
        body_vel: Float[torch.Tensor, "body_count 6 1"],
        lambda_n: Float[torch.Tensor, "body_count max_contacts 1"],
        lambda_t: Float[torch.Tensor, "body_count 2*max_contacts 1"],
        gamma: Float[torch.Tensor, "body_count max_contacts 1"],
        J_t: Float[torch.Tensor, "body_count 2*max_contacts 6"],
        contact_mask: Bool[torch.Tensor, "body_count max_contacts"],
        friction_coeff: Float[torch.Tensor, "body_count 1"],
    ):
        B = body_vel.shape[0]
        C = lambda_n.shape[1]

        # Split lambda_t into two parts
        lambda_t1 = lambda_t[:, :C, :]  # [B, C, 1]
        lambda_t2 = lambda_t[:, C:, :]  # [B, C, 1]
        J_t1 = J_t[:, :C, :]  # [B, C, 6]
        J_t2 = J_t[:, C:, :]  # [B, C, 6]

        active_mask = contact_mask.view(B, C, 1).float() # [B, C, 1]
        inactive_mask = 1 - active_mask # [B, C, 1]

        # Expand the friction coefficient
        mu = friction_coeff.unsqueeze(1).expand(B, C, 1)  # [B, C, 1]

        # Compute the friction impulse norm
        lambda_t_norm = torch.sqrt(lambda_t1 ** 2 + lambda_t2 ** 2 + self.eps)  # [B, C, 1]

        # ∂res_fr / ∂body_vel
        dres_fr1_dbody_vel = J_t1 * active_mask  # [B, C, 6]
        dres_fr2_dbody_vel = J_t2 * active_mask  # [B, C, 6]
        dres_fr_dbody_vel = torch.cat([dres_fr1_dbody_vel, dres_fr2_dbody_vel], dim=1)  # [B, 2*C, 6]

        # ∂res_fr / ∂lambda_n
        dres_fr_dlambda_n = torch.zeros((B, 2 * C, C)) # [B, 2 * C, C]

        # ∂res_fr / ∂lambda_t
        # For res_fr1 / ∂lambda_t1
        dres_fr1_dlambda_t1_act = gamma * (lambda_t2**2 + self.eps) / (lambda_t_norm**3 + self.eps)  # [B, C, 1]
        dres_fr1_dlambda_t1_inact = -torch.ones_like(lambda_t1)  # [B, C, 1]
        dres_fr1_dlambda_t1 = dres_fr1_dlambda_t1_act * active_mask + dres_fr1_dlambda_t1_inact * inactive_mask  # [B, C, 1]
        dres_fr1_dlambda_t1 = torch.diag_embed(dres_fr1_dlambda_t1.squeeze(-1))  # [B, C, C]

        # For res_fr1 / ∂lambda_t2
        dres_fr1_dlambda_t2 = -gamma * lambda_t1 * lambda_t2 / (lambda_t_norm**3 + self.eps) * active_mask  # [B, C, 1]
        dres_fr1_dlambda_t2 = torch.diag_embed(dres_fr1_dlambda_t2.squeeze(-1))  # [B, C, C]

        # For res_fr2 / ∂lambda_t2
        dres_fr2_dlambda_t2_act = gamma * (lambda_t1**2 + self.eps) / (lambda_t_norm**3 + self.eps)  # [B, C, 1]
        dres_fr2_dlambda_t2_inact = -torch.ones_like(lambda_t2)  # [B, C, 1]
        dres_fr2_dlambda_t2 = dres_fr2_dlambda_t2_act * active_mask + dres_fr2_dlambda_t2_inact * inactive_mask  # [B, C, 1]
        dres_fr2_dlambda_t2 = torch.diag_embed(dres_fr2_dlambda_t2.squeeze(-1))  # [B, C, C]

        # For res_fr2 / ∂lambda_t1
        dres_fr2_dlambda_t1 = -gamma * lambda_t1 * lambda_t2 / (lambda_t_norm**3 + self.eps) * active_mask  # [B, C, 1]
        dres_fr2_dlambda_t1 = torch.diag_embed(dres_fr2_dlambda_t1.squeeze(-1))  # [B, C, C]

        # Combine derivatives for res_fr1
        dres_fr1_dlambda_t = torch.cat([dres_fr1_dlambda_t1, dres_fr1_dlambda_t2], dim=2)  # [B, C, 2*C]

        # Combine derivatives for res_fr2
        dres_fr2_dlambda_t = torch.cat([dres_fr2_dlambda_t1, dres_fr2_dlambda_t2], dim=2)  # [B, C, 2*C]

        # Combine for complete ∂res_fr / ∂lambda_t
        dres_fr_dlambda_t = torch.cat([dres_fr1_dlambda_t, dres_fr2_dlambda_t], dim=1)  # [B, 2*C, 2*C]

        # ∂res_fr / ∂gamma
        dres_fr1_dgamma = (lambda_t1 / (lambda_t_norm + self.eps)) * active_mask  # [B, C, 1]
        dres_fr1_dgamma = torch.diag_embed(dres_fr1_dgamma.squeeze(-1))  # [B, C, C]

        dres_fr2_dgamma = (lambda_t2 / (lambda_t_norm + self.eps)) * active_mask  # [B, C, 1]
        dres_fr2_dgamma = torch.diag_embed(dres_fr2_dgamma.squeeze(-1))  # [B, C, C]

        dres_fr_dgamma = torch.cat([dres_fr1_dgamma, dres_fr2_dgamma], dim=1)  # [B, 2*C, C]

        # For friction cone constraint (NCP)
        dres_frc_dgamma_act, dres_frc_db_act = self.fb.derivatives(gamma, mu * lambda_n - lambda_t_norm)  # [B, C, 1]

        # ∂res_frc / ∂(mu * lambda_n - lambda_t_norm)
        dres_frc_db = dres_frc_db_act * active_mask  # [B, C, 1]

        # ∂res_frc / ∂body_vel
        dres_frc_dbody_vel = torch.zeros((B, C, 6), device=body_vel.device)  # [B, C, 6]

        # ∂res_frc / ∂lambda_n
        dres_frc_dlambda_n = torch.diag_embed((dres_frc_db * mu).squeeze(-1))  # [B, C, C]

        # ∂res_frc / ∂lambda_t
        dres_frc_dlambda_t1 = -dres_frc_db * lambda_t1 / (lambda_t_norm + self.eps)  # [B, C, 1]
        dres_frc_dlambda_t1 = torch.diag_embed(dres_frc_dlambda_t1.squeeze(-1))  # [B, C, C]

        dres_frc_dlambda_t2 = -dres_frc_db * lambda_t2 / (lambda_t_norm + self.eps)  # [B, C, 1]
        dres_frc_dlambda_t2 = torch.diag_embed(dres_frc_dlambda_t2.squeeze(-1))  # [B, C, C]

        dres_frc_dlambda_t = torch.cat([dres_frc_dlambda_t1, dres_frc_dlambda_t2], dim=2)  # [B, C, 2*C]

        # ∂res_frc / ∂gamma
        dres_frc_dgamma_inact = -torch.ones_like(gamma)  # [B, C, 1]
        dres_frc_dgamma = dres_frc_dgamma_act * active_mask + dres_frc_dgamma_inact * inactive_mask  # [B, C, 1]
        dres_frc_dgamma = torch.diag_embed(dres_frc_dgamma.squeeze(-1))  # [B, C, C]

        # Combine all derivatives
        # ∂res / ∂body_vel
        dres_dbody_vel = torch.cat([dres_fr_dbody_vel, dres_frc_dbody_vel], dim=1)  # [B, 3*C, 6]

        # ∂res / ∂lambda_n
        dres_dlambda_n = torch.cat([dres_fr_dlambda_n, dres_frc_dlambda_n], dim=1)  # [B, 3*C, C]

        # ∂res / ∂lambda_t
        dres_dlambda_t = torch.cat([dres_fr_dlambda_t, dres_frc_dlambda_t], dim=1)  # [B, 3*C, 2*C]

        # ∂res / ∂gamma
        dres_dgamma = torch.cat([dres_fr_dgamma, dres_frc_dgamma], dim=1)  # [B, 3*C, C]

        return dres_dbody_vel, dres_dlambda_n, dres_dlambda_t, dres_dgamma

# L ... number of all joints
# D ... maximum number of joints per body

class RevoluteConstraint:
    def __init__(self,
        model: Model,
        joint_parent: Int[torch.Tensor, "joint_count"],
        joint_child: Int[torch.Tensor, "joint_count"],
        joint_trans_p: Float[torch.Tensor, "joint_count 7 1"],
        joint_trans_c: Float[torch.Tensor, "joint_count 7 1"],
        max_joints: int,
        device: torch.device
    ):
        self.joint_parent = joint_parent
        self.joint_child = joint_child
        self.joint_trans_p = joint_trans_c
        self.joint_trans_c = joint_trans_p

        self.max_joints = max_joints

        self.device = device

    def _get_joint_frames(self,
        body_trans: Float[torch.Tensor, "body_count 7 1"]
    ) -> Tuple[Float[torch.Tensor, "joint_count 7 1"],
        Float[torch.Tensor, "joint_count 7 1"],
        Float[torch.Tensor, "joint_count 3 1"],
        Float[torch.Tensor, "joint_count 3 1"]]:
       B = body_trans.shape[0]
       L = self.joint_parent.shape[0]

       parent_transforms = body_trans[self.joint_parent] # [L, 7, 1]
       child_transforms = body_trans[self.joint_child] # [L, 7, 1]

       # Joint frame computed from the parent body
       X_p = transform_multiply_batch(parent_transforms, self.joint_trans_p)  # [L, 7, 1]

       # Joint frame computed from the child body
       X_c = transform_multiply_batch(child_transforms, self.joint_trans_c)  # [L, 7, 1]

       # Get the relative vector from the body frame to joint frame in world coordinates
       r_p = X_c[:, :3] - X_p[:, :3] # [L, 3, 1]
       r_c = X_p[:, :3] - X_c[:, :3] # [L, 3, 1]

       return X_p, X_c, r_p, r_c

    def _get_translational_constraints(self,
        X_p: Float[torch.Tensor, "joint_count 7 1"],
        X_c: Float[torch.Tensor, "joint_count 7 1"]
    ) -> Float[torch.Tensor, "joint_count 3 1"]:
        constraint_trans = X_p[:, :3] - X_c[:, :3] # [L, 3, 1]
        return constraint_trans

    def _get_rotational_constraints(self,
        X_p: Float[torch.Tensor, "joint_count 7 1"],
        X_c: Float[torch.Tensor, "joint_count 7 1"]
    ) -> Float[torch.Tensor, "joint_count 2 1"]:
        L = X_p.shape[0]

        # Create unit vectors for x, y, and z axes
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(L, 1)  # [L, 3]
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(L, 1)  # [L, 3]
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(L, 1)  # [L, 3]

        # Rotate unit vectors to child and parent frames
        x_axis_child = rotate_vectors_batch(x_axis.unsqueeze(-1), X_c[:, 3:])  # [L, 3, 1]
        y_axis_child = rotate_vectors_batch(y_axis.unsqueeze(-1), X_c[:, 3:])  # [L, 3, 1]
        z_axis_parent = rotate_vectors_batch(z_axis.unsqueeze(-1), X_p[:, 3:])  # [L, 3, 1]

        # Rotational constraints: x and y of child perpendicular to z of parent
        x_constraint = (x_axis_child * z_axis_parent).sum(dim=1).unsqueeze(1)  # [L, 1, 1]
        y_constraint = (y_axis_child * z_axis_parent).sum(dim=1).unsqueeze(1)  # [L, 1, 1]

        constraints_rot = torch.cat([x_constraint, y_constraint], dim=1)  # [L, 2, 1]

        return constraints_rot

    def get_values(self, body_trans: Float[torch.Tensor, "body_count 7 1"]):
        B = body_trans.shape[0]

        constraints = torch.zeros(B, self.max_joints, 5, 1, device=self.device) # [B, D, 5, 1]

        X_p, X_c, _, _ = self._get_joint_frames(body_trans) # [L, 7, 1], [L, 7, 1]

        constraint_trans_p = self._get_translational_constraints(X_p, X_c) # [L, 3, 1]
        constraint_rot_p = self._get_rotational_constraints(X_p, X_c) # [L, 2, 1]

        constraints_joints_p = torch.cat([constraint_trans_p, constraint_rot_p], dim=1) # [L, 5, 1]
        constraints_joints_c = -constraints_joints_p # [L, 5, 1]
        constraints_joints = torch.cat([constraints_joints_p, constraints_joints_c], dim=0) # [2L, 5, 1]

        all_body_indices = torch.cat([self.joint_parent, self.joint_child], dim=0)
        for b in range(B):
            body_indices = torch.where(all_body_indices == b)[0][:self.max_joints] # [body_joint_count]
            constraints[b, :self.max_joints] = constraints_joints[body_indices]

        return constraints

    def _get_translational_jacobian(self, X_p, X_c, r_p, r_c):
        L = X_p.shape[0]
        E = torch.eye(3, device=self.device).repeat(L, 1, 1) # [L, 3, 3]
        r_px = skew_symmetric_matrix_batch(r_p) # [L, 3, 3]
        r_cx = skew_symmetric_matrix_batch(r_c) # [L, 3, 3]

        J_p = torch.cat([r_px, -E], dim=2) # [L, 3, 6]
        J_c = torch.cat([-r_cx, E], dim=2) # [L, 3, 6]

        return J_p, J_c

    def _get_rotational_jacobian(self, X_p, X_c):
        L = X_p.shape[0]

        # Create unit vectors for x, y, and z axes
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(L, 1)  # [L, 3]
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(L, 1)  # [L, 3]
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(L, 1)  # [L, 3]

        # Rotate unit vectors to child and parent frames
        x_axis_child = rotate_vectors_batch(x_axis.unsqueeze(-1), X_c[:, 3:])  # [L, 3, 1]
        y_axis_child = rotate_vectors_batch(y_axis.unsqueeze(-1), X_c[:, 3:])  # [L, 3, 1]
        z_axis_parent = rotate_vectors_batch(z_axis.unsqueeze(-1), X_p[:, 3:])  # [L, 3, 1]

        x_c_cross_z_p = torch.linalg.cross(x_axis_child, z_axis_parent, dim=1) # [L, 3, 1]
        y_c_cross_z_p = torch.linalg.cross(y_axis_child, x_axis_child, dim=1) # [L, 3, 1]

        J_p = torch.zeros((L, 2, 6), device=self.device)
        J_p[:, 0, :3] = x_c_cross_z_p.squeeze(-1)
        J_p[:, 1, :3] = y_c_cross_z_p.squeeze(-1)

        J_c = torch.zeros((L, 2, 6), device=self.device)
        J_c[:, 0, :3] = x_c_cross_z_p.squeeze(-1)
        J_c[:, 1, :3] = y_c_cross_z_p.squeeze(-1)

        return J_p, J_c

    def _get_jacobian(self, body_trans: torch.Tensor):
        B = body_trans.shape[0]
        D = self.max_joints

        J = torch.zeros((B, D, 5, 6), device=self.device)

        X_p, X_c, r_p, r_c = self._get_joint_frames(body_trans)

        J_trans_p, J_trans_c = self._get_translational_jacobian(X_p, X_c, r_p, r_c)
        J_rot_p, J_rot_c = self._get_rotational_jacobian(X_p, X_c)

        J_trans_joints = torch.cat([J_trans_p, J_trans_c], dim=0) # [2L, 3, 6]
        J_rot_joints = torch.cat([J_rot_p, J_rot_c], dim=0) # [2L, 2, 6]
        all_body_indices = torch.cat([self.joint_parent, self.joint_child], dim=0) # [2L]

        for b in range(B):
            body_indices = torch.where(all_body_indices == b)[0][:self.max_joints]  # [num_joints]        constraints = C_joints[joint_indices]
            J[b, :self.max_joints, :3] = J_trans_joints[body_indices]
            J[b, :self.max_joints, 3:] = J_rot_joints[body_indices]

        return J




# def dC_revolute(
#
#     body_q: torch.Tensor,
#     body_qd: torch.Tensor,
#     joint_parent: torch.Tensor,
#     joint_child: torch.Tensor,
#     joint_X_p: torch.Tensor,
#     joint_X_c: torch.Tensor,
#     joint_basis: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     J_p_trans, J_c_trans = J_translation(
#         body_q, joint_parent, joint_child, joint_X_p, joint_X_c, joint_basis
#     )  # ([N, 3, 6], [N, 3, 6])
#     J_p_rot, J_c_rot = J_rotation(
#         body_q, joint_parent, joint_child, joint_X_p, joint_X_c, joint_basis
#     )  # ([N, 2, 6], [N, 2, 6])

#     # Stack the translation and rotation Jacobians
#     J_p = torch.cat([J_p_trans, J_p_rot], dim=1)  # [N, 5, 6]
#     J_c = torch.cat([J_c_trans, J_c_rot], dim=1)  # [N, 5, 6]

#     body_qd_p = body_qd[joint_parent]  # [N, 6]
#     body_qd_c = body_qd[joint_child]  # [N, 6]
#     dC_p = (J_p * body_qd_p.unsqueeze(1)).sum(dim=2)  # [N, 5]
#     dC_c = (J_c * body_qd_c.unsqueeze(1)).sum(dim=2)  # [N, 5]
#     return dC_p, dC_c


# def J_translation(
#     body_q: torch.Tensor,
#     joint_parent: torch.Tensor,
#     joint_child: torch.Tensor,
#     joint_X_p: torch.Tensor,
#     joint_X_c: torch.Tensor,
#     joint_basis: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     device = body_q.device
#     joint_count = joint_parent.shape[0]
#     E = torch.eye(3, device=device).repeat(joint_count, 1, 1)  # [N, 3, 3]

#     parent_q = body_q[joint_parent]  # [N, 4]
#     child_q = body_q[joint_child]  # [N, 4]

#     # Joint frame from the parent body
#     X_wj_p = transform_multiply_batch(parent_q, joint_X_p)  # [N, 7]
#     X_wj_c = transform_multiply_batch(child_q, joint_X_c)  # [N, 7]

#     r_p = X_wj_p[:, :3] - parent_q[:, :3]  # [N, 3]
#     r_c = X_wj_c[:, :3] - child_q[:, :3]  # [N, 3]

#     r_px = skew_symmetric_matrix_batch(r_p)  # [N, 3, 3]
#     r_cx = skew_symmetric_matrix_batch(r_c)  # [N, 3, 3]

#     J_p = torch.cat([r_px, -E], dim=1)  # [N, 3, 6]
#     J_c = torch.cat([r_cx, E], dim=1)  # [N, 3, 6]

#     return J_p, J_c


# def J_rotation(
#     body_q: torch.Tensor,
#     joint_parent: torch.Tensor,
#     joint_child: torch.Tensor,
#     joint_X_p: torch.Tensor,
#     joint_X_c: torch.Tensor,
#     joint_basis: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     device = body_q.device
#     joint_count = joint_parent.shape[0]
#     parent_q = body_q[joint_parent]  # [N, 4]
#     child_q = body_q[joint_child]  # [N, 4]

#     X_wj_p = transform_multiply_batch(parent_q, joint_X_p)  # [N, 7]
#     X_wj_c = transform_multiply_batch(child_q, joint_X_c)  # [N, 7]

#     x_axis = torch.tensor([1.0, 0.0, 0.0], device=device).repeat(
#         joint_count, 1
#     )  # [N, 3]
#     y_axis = torch.tensor([0.0, 1.0, 0.0], device=device).repeat(
#         joint_count, 1
#     )  # [N, 3]
#     z_axis = torch.tensor([0.0, 0.0, 1.0], device=device).repeat(
#         joint_count, 1
#     )  # [N, 3]

#     x_axis_child = rotate_vectors_batch(x_axis, X_wj_c)  # [N, 3]
#     y_axis_child = rotate_vectors_batch(y_axis, X_wj_c)  # [N, 3]
#     z_axis_parent = rotate_vectors_batch(z_axis, X_wj_p)  # [N, 3]

#     x_c_x_z_p = torch.linalg.cross(x_axis_child, z_axis_parent, dim=1)  # [N, 3]
#     y_c_x_z_p = torch.linalg.cross(y_axis_child, z_axis_parent, dim=1)  # [N, 3]

#     J_p = torch.zeros(joint_count, 2, 6, device=device)  # [N, 2, 6]
#     J_p[:, 0, :3] = -x_c_x_z_p
#     J_p[:, 1, :3] = -y_c_x_z_p

#     J_c = torch.zeros(joint_count, 2, 6, device=device)  # [N, 2, 6]
#     J_c[:, 0, :3] = x_c_x_z_p
#     J_c[:, 1, :3] = y_c_x_z_p

#     return J_p, J_c
