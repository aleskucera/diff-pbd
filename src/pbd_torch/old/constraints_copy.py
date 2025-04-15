from typing import Tuple

import torch
from jaxtyping import Float, Bool, Int
from pbd_torch.transform import rotate_vectors_batch
from pbd_torch.transform import transform_multiply_batch
from pbd_torch.transform import transform_points_batch
from pbd_torch.ncp import ScaledFisherBurmeister
from pbd_torch.model import Model


def skew_symmetric_matrix_batch(
    vectors: Float[torch.Tensor, "B D 3 1"]
) -> Float[torch.Tensor, "B D 3 3"]:
    """
    Compute the skew-symmetric matrix of a batch of vectors.

    Args:
        vectors: Tensor of shape [N, 3, 1].

    Returns:
        Tensor of shape [N, 3, 3] with skew-symmetric matrices.
    """

    N = vectors.shape[0]
    D = vectors.shape[1]
    skew = torch.zeros(N, D, 3, 3, device=vectors.device, dtype=vectors.dtype)
    skew[..., 0, 1] = -vectors[..., 2, 0]
    skew[..., 0, 2] = vectors[..., 1, 0]
    skew[..., 1, 0] = vectors[..., 2, 0]
    skew[..., 1, 2] = -vectors[..., 0, 0]
    skew[..., 2, 0] = -vectors[..., 1, 0]
    skew[..., 2, 1] = vectors[..., 0, 0]
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
        lambda_j: Float[torch.Tensor, "body_count 10 * max_joints 1"],
        body_f: Float[torch.Tensor, "body_count 6 1"],
        J_n: Float[torch.Tensor, "body_count max_contacts 6"],
        J_t: Float[torch.Tensor, "body_count 2 * max_contacts 6"],
        J_j: Float[torch.Tensor, "body_count 5 * max_joints 6"],
        dt: float,
    ) -> Float[torch.Tensor, "body_count 6 1"]:
        return (
            torch.matmul(self.mass_matrix, (body_vel - body_vel_prev))  # M @ (v - v_prev)
            - torch.matmul(J_n.transpose(2, 1), lambda_n)  # J_n^T @ λ_n
            - torch.matmul(J_t.transpose(2, 1), lambda_t)  # J_t^T @ λ_t
            - torch.matmul(J_j.transpose(2, 1), lambda_j)  # J_j^T @ λ_j
            - body_f * dt  # f·dt
            - torch.matmul(self.mass_matrix, self.g_accel) * dt  # M @ g·dt
        )

    def get_derivatives(self,
         J_n: Float[torch.Tensor, "body_count max_contacts 6"],
         J_t: Float[torch.Tensor, "body_count 2 * max_contacts 6"],
         J_j: Float[torch.Tensor, "body_count 5 * max_joints 6"],
         ) -> Tuple[Float[torch.Tensor, "body_count 6 6"],
            Float[torch.Tensor, "body_count max_contacts 6"],
            Float[torch.Tensor, "body_count 2 * max_contacts 6"],
            torch.Tensor]:
        # ∂res/∂body_vel
        dres_dbody_vel = self.mass_matrix # [B, 6, 6]

        # ∂res/∂lambda_n
        dres_dlambda_n = -J_n.transpose(1, 2) # [B, 6, C]

        # ∂res/∂lambda_t
        dres_dlambda_t = -J_t.transpose(1, 2) # [B, 6, 2 * C]

        # ∂res/∂lambda_j
        dres_dlambda_j = -J_j.transpose(1, 2) # [B, 6, 10 * D]

        return dres_dbody_vel, dres_dlambda_n, dres_dlambda_t, dres_dlambda_j

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

class RevoluteConstraint:
    def __init__(self,
        model: Model,
        device: torch.device
    ):
        self.joint_trans = None # [B, D, 7, 1]
        self.joint_trans_other = None # [B, D, 7, 1]
        self.joint_map = None # [B, D]
        self.joint_mask = None  # [B, D]
        self.joint_mask_inv = None # [B, D]

        self.stabilization_factor = 0.0

        self.device = device

        self._get_joint_attributes(model)

    def _get_joint_attributes(self, model: Model):
        B = model.body_q.shape[0] # number of bodies

        self.joint_trans = torch.zeros((B, 0, 7, 1), device=self.device) # [B, D, 7, 1]
        self.joint_trans_other = torch.zeros((B, 0, 7, 1), device=self.device)  # [B, D, 7, 1]
        self.joint_mask = torch.zeros((B, 0), device=self.device, dtype=torch.bool) # [B, D]
        self.joint_map = torch.zeros((B, 0), device=self.device, dtype=torch.long) # [B, D]

        for i in range(model.joint_count):
            parent_idx = model.joint_parent[i]
            child_idx = model.joint_child[i]
            X_p = model.joint_X_p[i]
            X_c = model.joint_X_c[i]

            # ------------------ PARENT ------------------
            parent_transforms = self.joint_trans[parent_idx]

            # Check is there is a slot for additional transforms for parent
            is_zero = torch.all(parent_transforms == 0, dim=1) # [D, 1]
            zero_indices = torch.where(is_zero)[0] # indices, where the transform is zero

            if len(zero_indices) == 0:
                self.joint_trans = torch.cat([self.joint_trans, torch.zeros((B, 1, 7, 1), device=self.device)], dim=1) # [B, D, 7, 1]
                self.joint_trans_other = torch.cat([self.joint_trans_other, torch.zeros((B, 1, 7, 1), device=self.device)], dim=1) # [B, D, 7, 1]
                self.joint_mask = torch.cat([self.joint_mask, torch.zeros((B, 1), device=self.device)], dim=1) # [B, D]
                self.joint_map = torch.cat([self.joint_map, torch.zeros((B, 1), device=self.device, dtype=torch.long)], dim=1) # [B, D]

                slot_idx = self.joint_trans.shape[1] - 1
            else:
                slot_idx = zero_indices[0]

            self.joint_trans[parent_idx, slot_idx] = X_p
            self.joint_trans_other[parent_idx, slot_idx] = X_c
            self.joint_mask[parent_idx, slot_idx] = True
            self.joint_map[parent_idx, slot_idx] = child_idx

            # ------------------ CHILD ------------------
            child_transforms = self.joint_trans[child_idx]

            # Check is there is a slot for additional transforms for child
            is_zero = torch.all(child_transforms == 0, dim=1) # [D, 1]
            zero_indices = torch.where(is_zero)[0] # indices, where the transform is zero

            if len(zero_indices) == 0:
                self.joint_trans = torch.cat([self.joint_trans, torch.zeros((B, 1, 7, 1), device=self.device)], dim=1) # [B, D, 7, 1]
                self.joint_trans_other = torch.cat([self.joint_trans_other, torch.zeros((B, 1, 7, 1), device=self.device)], dim=1) # [B, D, 7, 1]
                self.joint_mask = torch.cat([self.joint_mask, torch.zeros((B, 1), device=self.device)], dim=1) # [B, D]
                self.joint_map = torch.cat([self.joint_map, torch.zeros((B, 1), device=self.device, dtype=torch.long)], dim=1) # [B, D]

                slot_idx = self.joint_trans.shape[1] - 1
            else:
                slot_idx = zero_indices[0]

            self.joint_trans[child_idx, slot_idx] = X_c
            self.joint_trans_other[child_idx, slot_idx] = X_p
            self.joint_mask[child_idx, slot_idx] = True
            self.joint_map[child_idx, slot_idx] = parent_idx

        return self.joint_trans, self.joint_trans_other, self.joint_mask

    def _get_joint_frames(self,
        body_trans: Float[torch.Tensor, "body_count 7 1"]
    ) -> Tuple[Float[torch.Tensor, "body_count max_joints 7 1"],
        Float[torch.Tensor, "body_count max_joints 7 1"],
        Float[torch.Tensor, "body_count max_joints 3 1"]]:
       B = body_trans.shape[0]
       D = self.joint_trans.shape[1]

       transforms = body_trans.unsqueeze(1).expand(B, D, 7, 1)
       transforms_other = body_trans[self.joint_map] # [B, D, 7, 1]

       # Joint frame computed from the parent and child bodies
       X = transform_multiply_batch(transforms, self.joint_trans)  # [B, D, 7, 1]
       X_other = transform_multiply_batch(transforms_other, self.joint_trans_other)  # [B, D, 7, 1]

       # Get the relative vector from the body frame to joint frame in world coordinates
       r = X[:, :, :3] - transforms[:, :, :3] # [B, D, 3, 1]

       return X, X_other, r

    def _get_translational_errors(self,
        X: Float[torch.Tensor, "body_count max_joints 7 1"],
        X_other: Float[torch.Tensor, "body_count max_joints 7 1"]
    ) -> Tuple[Float[torch.Tensor, "body_count max_joints 1"],
        Float[torch.Tensor, "body_count max_joints 1"],
        Float[torch.Tensor, "body_count max_joints 1"]]:
        res_x = X_other[:, :, 0] - X[:, :, 0] # [B, D, 1]
        res_y = X_other[:, :, 1] - X[:, :, 1] # [B, D, 1]
        res_z = X_other[:, :, 2] - X[:, :, 2] # [B, D, 1]

        return res_x, res_y, res_z

    def _get_rotational_errors(self,
        X: Float[torch.Tensor, "body_count max_joints 7 1"],
        X_peer: Float[torch.Tensor, "body_count max_joints 7 1"]
    ) -> Tuple[Float[torch.Tensor, "body_count max_joints 1"],
        Float[torch.Tensor, "body_count max_joints 1"]]:
        B = X.shape[0]
        D = X.shape[1]

        # Create unit vectors for x, y, and z axes
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(B, D, 1)  # [B, D, 3]
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(B, D, 1)  # [B, D, 3]
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(B, D, 1)  # [B, D, 3]

        # Rotate unit vectors to child and parent frames
        x_axis_self = rotate_vectors_batch(x_axis.unsqueeze(-1), X[:, :, 3:]) # [B, D, 3, 1]
        y_axis_self = rotate_vectors_batch(y_axis.unsqueeze(-1), X[:, :, 3:]) # [B, D, 3, 1]
        z_axis_peer = rotate_vectors_batch(z_axis.unsqueeze(-1), X_peer[:, :, 3:]) # [B, D, 3, 1]

        # Rotational constraints: x and y of child perpendicular to z of parent
        res_x = torch.matmul(x_axis_self.transpose(2, 3), z_axis_peer).squeeze(-1)  # [B, D, 1]
        res_y = torch.matmul(y_axis_self.transpose(2, 3), z_axis_peer).squeeze(-1)  # [B, D, 1]

        return res_x, res_y

    def get_errors(self,
        body_trans: Float[torch.Tensor, "body_count 7 1"]
    ) -> Float[torch.Tensor, "body_count max_joints 5 1"]:

        mask = torch.cat([
            self.joint_mask,
            self.joint_mask,
            self.joint_mask,
            self.joint_mask,
            self.joint_mask], dim=1).unsqueeze(-1) # [B, 5 * D, 1]

        X, X_peer, _ = self._get_joint_frames(body_trans) # [B, D, 7, 1], [B, D, 7, 1]

        err_tx, err_ty, err_tz = self._get_translational_errors(X, X_peer) # [B, D, 1], [B, D, 1], [B, D, 1]

        err_rx, err_ry = self._get_rotational_errors(X, X_peer) # [B, D, 1], [B, D, 1]

        errors = torch.cat([err_tx, err_ty, err_tz, err_rx, err_ry], dim=1) # [B, 5 * D, 1]
        errors = errors * mask # [B, 5 * D, 1]

        return errors

    def _get_translational_jacobian(self,
        X: Float[torch.Tensor, "body_count max_joints 7 1"],
        r: Float[torch.Tensor, "body_count max_joints 3 1"],
    ) -> Float[torch.Tensor, "body_count 3 * max_joints 6"]:
        B = X.shape[0]
        D = X.shape[1]

        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(B, D, 1)  # [B, D, 3]
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(B, D, 1)  # [B, D, 3]
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(B, D, 1)  # [B, D, 3]

        r_skew = skew_symmetric_matrix_batch(r) # [B, D, 3, 3]

        r_skew_x = r_skew[:, :, :, 0] # [B, D, 3]
        r_skew_y = r_skew[:, :, :, 1] # [B, D, 3]
        r_skew_z = r_skew[:, :, :, 2] # [B, D, 3]

        Jx = torch.cat([-r_skew_x, x_axis], dim=2) # [B, D, 6]
        Jy = torch.cat([-r_skew_y, y_axis], dim=2) # [B, D, 6]
        Jz = torch.cat([-r_skew_z, z_axis], dim=2) # [B, D, 6]

        J = torch.cat([Jx, Jy, Jz], dim=1) # [B, 3 * D, 6]

        return J

    def _get_rotational_jacobian(self,
        X: Float[torch.Tensor, "body_count max_joints 7 1"],
        X_peer: Float[torch.Tensor, "body_count max_joints 7 1"],
    ) -> Float[torch.Tensor, "body_count 2 * max_joints 6"]:
        B = X.shape[0]
        D = X.shape[1]

        # Create unit vectors for x, y, and z axes
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(B, D, 1) # [B, D, 3]
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(B, D, 1) # [B, D, 3]
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(B, D, 1) # [B, D, 3]

        # Rotate unit vectors to child and parent frames
        x_axis_self = rotate_vectors_batch(x_axis.unsqueeze(-1), X[:, :, 3:])  # [B, D, 3, 1]
        y_axis_self = rotate_vectors_batch(y_axis.unsqueeze(-1), X[:, :, 3:])  # [B, D, 3, 1]
        z_axis_peer = rotate_vectors_batch(z_axis.unsqueeze(-1), X_peer[:, :, 3:])  # [B, D, 3, 1]

        x_cross_z = torch.linalg.cross(x_axis_self, z_axis_peer, dim=2).squeeze(-1) # [B, D, 3]
        y_cross_z = torch.linalg.cross(y_axis_self, z_axis_peer, dim=2).squeeze(-1) # [B, D, 3]

        zeros = torch.zeros((B, D, 3), device=self.device, dtype=x_cross_z.dtype) # [B, D, 3]

        Jx = torch.cat([x_cross_z, zeros], dim=2) # [B, D, 6]
        Jy = torch.cat([y_cross_z, zeros], dim=2) # [B, D, 6]
        J = torch.cat([Jx, Jy], dim=1) # [B, 2 * D, 6]

        return J

    def compute_jacobians(self,
        body_trans: Float[torch.Tensor, "body_count 7 1"]
    ) -> Float[torch.Tensor, "body_count 5 * max_joints 6"]:
        X, X_peer, r = self._get_joint_frames(body_trans)

        # Get translational Jacobians (3 constraints per joint)
        Jt = self._get_translational_jacobian(X, r)

        # Get rotational Jacobians (2 constraints per joint)
        Jr = self._get_rotational_jacobian(X, X_peer)

        # Combine translational and rotational Jacobians
        J = torch.cat([Jt, Jr], dim=1)  # [B, 5*D, 6]

        return J

    def get_residuals(self,
        body_vel: Float[torch.Tensor, "body_count 5 * max_joints 1"],
        lambda_j: Float[torch.Tensor, "body_count 5 * max_joints 1"],
        body_trans: Float[torch.Tensor, "body_count 7 1"],
        J_j: Float[torch.Tensor, "body_count 5 * max_joints 6"],
        dt: float
    ) -> Float[torch.Tensor, "body_count 5 * max_joints 1"]:
        """Compute the velocity-level residuals with Baumgarte stabilization.

        Args:
            body_vel: Parent body velocity [B, 6, 1]
            body_trans: Current body transforms [B, 7, 1]
            J_j: Jacobian matrix [B, 5*D, 6]
            dt: Time step

        Returns:
            Residual vector [B, 5*D, 1]
        """

        act_mask = torch.cat([
            self.joint_mask,
            self.joint_mask,
            self.joint_mask,
            self.joint_mask,
            self.joint_mask], dim=1).unsqueeze(-1) # [B, 5 * D, 1]
        act_mask_inv = 1 - act_mask # [B, 5 * D, 1]

        # Get position-level errors
        errors = self.get_errors(body_trans)

        # Compute velocity-level residuals with Baumgarte stabilization
        v_j = torch.matmul(J_j, body_vel)  # [B, 5*D, 6] @ [B, 6, 1] -> [B, 5*D, 1]
        bias = -(self.stabilization_factor / dt) * errors # [B, 5*D, 1]
        res_act = v_j + bias # [B, 5*D, 1]
        res_inact = -lambda_j # [B, 5*D, 1]
        res = res_act * act_mask + res_inact * act_mask_inv

        return res

    def get_derivatives(self,
        J_j: Float[torch.Tensor, "body_count 5 * max_joints 6"],
    ) -> Tuple[Float[torch.Tensor, "body_count 5 * max_joints 6"],
        torch.Tensor]:
        B = J_j.shape[0]
        D = J_j.shape[1] // 5

        act_mask = torch.cat([
            self.joint_mask,
            self.joint_mask,
            self.joint_mask,
            self.joint_mask,
            self.joint_mask], dim=1).unsqueeze(-1) # [B, 5 * D, 1]
        act_mask_inv = 1 - act_mask

        dres_dbody_vel_act = J_j # [B, 5*D, 6]
        dres_dbody_vel_inact = torch.zeros_like(J_j) # [B, 5*D, 6]
        dres_dbody_vel = dres_dbody_vel_act * act_mask + dres_dbody_vel_inact * act_mask_inv

        dres_dlambda_j_act = torch.zeros((B, 5 * D), device=self.device, dtype=torch.float32) # [B, 5*D]
        dres_dlambda_j_inact = -torch.ones((B, 5 * D), device=self.device, dtype=torch.float32) # [B, 5*D]
        dres_dlambda_j = dres_dlambda_j_act * act_mask.squeeze(-1) + dres_dlambda_j_inact * act_mask_inv.squeeze(-1)
        dres_dlambda_j = torch.diag_embed(dres_dlambda_j) # [B, 5*D, 5*D]

        return dres_dbody_vel, dres_dlambda_j
