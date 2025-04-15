from typing import Tuple

import torch
from jaxtyping import Float, Bool, Int
from pbd_torch.transform import rotate_vectors_batch
from pbd_torch.transform import transform_multiply_batch
from pbd_torch.transform import transform_points_batch
from pbd_torch.ncp import ScaledFisherBurmeister
from pbd_torch.model import Model


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
        joint_trans_parent: Float[torch.Tensor, "max_joints 7 1"],
        joint_trans_child: Float[torch.Tensor, "max_joints 7 1"],
        joint_mask: Bool[torch.Tensor, "body_count max_joints"],
        device: torch.device
    ):
        self.joint_mask = joint_mask.float()  # [B, D]
        self.trans_parent = joint_trans_child  # [B, D, 7, 1]
        self.trans_child = joint_trans_parent  # [B, D, 7, 1]

        self.stabilization_factor = 0.2

        self.device = device

    def _get_joint_frames(self,
        body_trans: Float[torch.Tensor, "body_count 7 1"]
    ) -> Tuple[Float[torch.Tensor, "body_count max_joints 7 1"],
        Float[torch.Tensor, "body_count max_joints 7 1"],
        Float[torch.Tensor, "body_count max_joints 3 1"],
        Float[torch.Tensor, "body_count max_joints 3 1"]]:
       B = body_trans.shape[0]
       D = self.joint_mask.shape[1]

       transforms = body_trans.unsqueeze(1).expand(B, D, 7, 1)

       # Joint frame computed from the parent and child bodies
       X_p = transform_multiply_batch(transforms, self.trans_parent)  # [B, D, 7, 1]
       X_c = transform_multiply_batch(transforms, self.trans_child)  # [B, D, 7, 1]

       # Get the relative vector from the body frame to joint frame in world coordinates
       r_p = X_c[:, :, :3] - X_p[:, :, :3] # [B, D, 3, 1]
       r_c = X_p[:, :, :3] - X_c[:, :, :3] # [B, D, 3, 1]

       return X_p, X_c, r_p, r_c

    def _get_translational_errors(self,
        X_p: Float[torch.Tensor, "body_count max_joints 7 1"],
        X_c: Float[torch.Tensor, "body_count max_joints 7 1"]
    ) -> Tuple[Float[torch.Tensor, "body_count max_joints 1"],
        Float[torch.Tensor, "body_count max_joints 1"],
        Float[torch.Tensor, "body_count max_joints 1"]]:
        res_x = X_p[:, :, 0] - X_c[:, :, 0] # [B, D, 1]
        res_y = X_p[:, :, 1] - X_c[:, :, 1] # [B, D, 1]
        res_z = X_p[:, :, 2] - X_c[:, :, 2] # [B, D, 1]

        return res_x, res_y, res_z

    def _get_rotational_errors(self,
        X_p: Float[torch.Tensor, "body_count max_joints 7 1"],
        X_c: Float[torch.Tensor, "body_count max_joints 7 1"]
    ) -> Tuple[Float[torch.Tensor, "body_count max_joints 1"],
        Float[torch.Tensor, "body_count max_joints 1"]]:
        B = X_p.shape[0]
        D = X_p.shape[1]

        # Create unit vectors for x, y, and z axes
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(B, D, 1)  # [B, D, 3]
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(B, D, 1)  # [B, D, 3]
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(B, D, 1)  # [B, D, 3]

        # Rotate unit vectors to child and parent frames
        x_axis_child = rotate_vectors_batch(x_axis.unsqueeze(-1), X_c[:, :, :3]) # [B, D, 3, 1]
        y_axis_child = rotate_vectors_batch(y_axis.unsqueeze(-1), X_c[:, :, :3]) # [B, D, 3, 1]
        z_axis_parent = rotate_vectors_batch(z_axis.unsqueeze(-1), X_p[:, :, :3]) # [B, D, 3, 1]

        # Rotational constraints: x and y of child perpendicular to z of parent
        res_x = torch.matmul(x_axis_child.transpose(2, 3), z_axis_parent).squeeze(-1)  # [B, D, 1]
        res_y = torch.matmul(y_axis_child.transpose(2, 3), z_axis_parent).squeeze(-1)  # [B, D, 1]

        return res_x, res_y

    def get_errors(self,
        body_trans: Float[torch.Tensor, "body_count 7 1"]
    ) -> Float[torch.Tensor, "body_count max_joints 5 1"]:

        X_p, X_c, _, _ = self._get_joint_frames(body_trans) # [B, D, 7, 1], [B, D, 7, 1]

        err_tx_p_act, err_ty_p_act, err_tz_p_act = self._get_translational_errors(X_p, X_c) # [B, D, 1], [B, D, 1], [B, D, 1]
        err_tx_p = err_tx_p_act * self.joint_mask # [B, D, 1]
        err_ty_p = err_ty_p_act * self.joint_mask # [B, D, 1]
        err_tz_p = err_tz_p_act * self.joint_mask # [B, D, 1]

        err_rx_p_act, err_ry_p_act = self._get_rotational_errors(X_p, X_c) # [B, D, 1], [B, D, 1]
        err_rx_p = err_rx_p_act * self.joint_mask # [B, D, 1]
        err_ry_p = err_ry_p_act * self.joint_mask # [B, D, 1]

        errors_p = torch.cat([err_tx_p, err_ty_p, err_tz_p, err_rx_p, err_ry_p], dim=1) # [B, 5 * D, 1]
        errors_c = -errors_p # [B, 5 * D, 1]
        errors = torch.cat([errors_p, errors_c], dim=1) # [B, 10 * D, 1]

        return errors

    def _get_translational_jacobian(self,
        X_p: Float[torch.Tensor, "body_count max_joints 7 1"],
        X_c: Float[torch.Tensor, "body_count max_joints 7 1"],
        r_p: Float[torch.Tensor, "body_count max_joints 3 1"],
        r_c: Float[torch.Tensor, "body_count max_joints 3 1"]
    ) -> Tuple[Float[torch.Tensor, "body_count 3 * max_joints 6"],
        Float[torch.Tensor, "body_count 3 * max_joints 6"]]:
        B = X_p.shape[0]
        D = X_p.shape[1]

        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(B, D, 1)  # [B, D, 3]
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(B, D, 1)  # [B, D, 3]
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(B, D, 1)  # [B, D, 3]

        r_p_skew = skew_symmetric_matrix_batch(r_p) # [B, D, 3, 3]
        r_c_skew = skew_symmetric_matrix_batch(r_c) # [B, 3, 3]

        r_p_skew_x = r_p_skew[:, :, :, 0] # [B, D, 3]
        r_p_skew_y = r_p_skew[:, :, :, 1] # [B, D, 3]
        r_p_skew_z = r_p_skew[:, :, :, 2] # [B, D, 3]

        r_c_skew_x = r_c_skew[:, :, :, 0] # [B, D, 3]
        r_c_skew_y = r_c_skew[:, :, :, 1] # [B, D, 3]
        r_c_skew_z = r_c_skew[:, :, :, 2] # [B, D, 3]

        Jx_p = torch.cat([r_p_skew_x, -x_axis], dim=2) # [B, D, 6]
        Jy_p = torch.cat([r_p_skew_y, -y_axis], dim=2) # [B, D, 6]
        Jz_p = torch.cat([r_p_skew_z, -z_axis], dim=2) # [B, D, 6]

        J_p = torch.cat([Jx_p, Jy_p, Jz_p], dim=1) # [B, 3 * D, 6]


        Jx_c = torch.cat([-r_c_skew_x, x_axis], dim=2) # [B, D, 6]
        Jy_c = torch.cat([-r_c_skew_y, y_axis], dim=2) # [B, D, 6]
        Jz_c = torch.cat([-r_c_skew_z, z_axis], dim=2) # [B, D, 6]

        J_c = torch.cat([Jx_c, Jy_c, Jz_c], dim=1) # [B, 3 * D, 6]

        return J_p, J_c

    def _get_rotational_jacobian(self,
        X_p: Float[torch.Tensor, "body_count max_joints 7 1"],
        X_c: Float[torch.Tensor, "body_count max_joints 7 1"],
        ):
        B = X_p.shape[0]
        D = X_p.shape[1]

        # Create unit vectors for x, y, and z axes
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(B, D, 1) # [B, D, 3]
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(B, D, 1) # [B, D, 3]
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(B, D, 1) # [B, D, 3]

        # Rotate unit vectors to child and parent frames
        x_axis_child = rotate_vectors_batch(x_axis.unsqueeze(-1), X_c[:, :, 3:])  # [B, D, 3, 1]
        y_axis_child = rotate_vectors_batch(y_axis.unsqueeze(-1), X_c[:, :, 3:])  # [B, D, 3, 1]
        z_axis_parent = rotate_vectors_batch(z_axis.unsqueeze(-1), X_p[:, :, 3:])  # [B, D, 3, 1]

        x_c_cross_z_p = torch.linalg.cross(x_axis_child, z_axis_parent, dim=2).squeeze(-1) # [B, D, 3]
        y_c_cross_z_p = torch.linalg.cross(y_axis_child, x_axis_child, dim=2).squeeze(-1) # [B, D, 3]

        zeros = torch.zeros((B, D, 3), device=self.device, dtype=x_c_cross_z_p.dtype) # [B, D, 3]

        Jx_p = torch.cat([-x_c_cross_z_p, zeros], dim=2) # [B, D, 6]
        Jy_p = torch.cat([-y_c_cross_z_p, zeros], dim=2) # [B, D, 6]
        J_p = torch.cat([Jx_p, Jy_p], dim=1) # [B, 2 * D, 6]

        Jx_c = torch.cat([x_c_cross_z_p, zeros], dim=2) # [B, D, 6]
        Jy_c = torch.cat([y_c_cross_z_p, zeros], dim=2) # [B, D, 6]
        J_c = torch.cat([Jx_c, Jy_c], dim=1) # [B, 2 * D, 6]

        return J_p, J_c

    def get_jacobians(self,
        body_trans: Float[torch.Tensor, "body_count 7 1"]
    ) -> Float[torch.Tensor, "body_count 10 * max_joints 6"]:
        """Compute the Jacobians for both parent and child bodies.

        Returns:
            Tuple of (J_p, J_c) where each is [B, 5*D, 6]
        """
        X_p, X_c, r_p, r_c = self._get_joint_frames(body_trans)

        # Get translational Jacobians (3 constraints per joint)
        Jt_p, Jt_c = self._get_translational_jacobian(X_p, X_c, r_p, r_c)

        # Get rotational Jacobians (2 constraints per joint)
        Jr_p, Jr_c = self._get_rotational_jacobian(X_p, X_c)

        # Combine translational and rotational Jacobians
        J_p = torch.cat([Jt_p, Jr_p], dim=1)  # [B, 5*D, 6]
        J_c = torch.cat([Jt_c, Jr_c], dim=1)  # [B, 5*D, 6]
        J = torch.cat([J_p, J_c], dim=1)  # [B, 10*D, 6]

        return J

    def get_residuals(self,
        body_vel: Float[torch.Tensor, "body_count 10 * max_joints 1"],
        body_trans: Float[torch.Tensor, "body_count 7 1"],
        J_j: Float[torch.Tensor, "body_count 10 * max_joints 6"],
        dt: float
    ) -> Float[torch.Tensor, "body_count 10 * max_joints 1"]:
        """Compute the velocity-level residuals with Baumgarte stabilization.

        Args:
            body_vel: Parent body velocity [B, 6, 1]
            body_trans: Current body transforms [B, 7, 1]
            J_j: Jacobian matrix [B, 10*D, 6]
            dt: Time step

        Returns:
            Residual vector [B, 10*D, 1]
        """

        # Get position-level errors
        errors = self.get_errors(body_trans)

        # Compute velocity-level residuals with Baumgarte stabilization
        v_j = torch.matmul(J_j, body_vel)  # [B, 10*D, 6] @ [B, 6, 1] -> [B, 10*D, 1]
        bias = (self.stabilization_factor / dt) * errors # [B, 10*D, 1]
        res = v_j + bias # [B, 10*D, 1]

        return res

    def get_derivatives(self,
        body_vel: Float[torch.Tensor, "body_count 6 1"],
        body_trans: Float[torch.Tensor, "body_count 7 1"],
        J_j: Float[torch.Tensor, "body_count 10 * max_joints 6"],
        dt: float
    ) -> Float[torch.Tensor, "body_count 10 * max_joints 6"]:
        dres_dbody_vel = J_j
        return dres_dbody_vel
