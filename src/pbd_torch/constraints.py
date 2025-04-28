from typing import Tuple

import torch
from jaxtyping import Float, Bool
from pbd_torch.transform import rotate_vectors_batch
from pbd_torch.transform import transform_multiply_batch
from pbd_torch.transform import transform_points_batch
from pbd_torch.ncp import ScaledFisherBurmeister
from pbd_torch.model import Model


def skew_symmetric_matrix_batch(
    vectors: Float[torch.Tensor, "D 3 1"]
) -> Float[torch.Tensor, "D 3 3"]:
    """
    Compute the skew-symmetric matrix of a batch of vectors.

    Args:
        vectors: Tensor of shape [D, 3, 1].

    Returns:
        Tensor of shape [D, 3, 3] with skew-symmetric matrices.
    """

    D = vectors.shape[0]
    skew = torch.zeros(D, 3, 3, device=vectors.device, dtype=vectors.dtype)
    skew[..., 0, 1] = -vectors[..., 2, 0]
    skew[..., 0, 2] = vectors[..., 1, 0]
    skew[..., 1, 0] = vectors[..., 2, 0]
    skew[..., 1, 2] = -vectors[..., 0, 0]
    skew[..., 2, 0] = -vectors[..., 1, 0]
    skew[..., 2, 1] = vectors[..., 0, 0]
    return skew

class DynamicsConstraint:
    def __init__(self, model: Model):
        self.device = model.device
        self.mass_matrix = model.mass_matrix
        self.g_accel = model.g_accel.expand(self.mass_matrix.shape[0], 6, 1)
        self.joint_parent = model.joint_parent
        self.joint_child = model.joint_child

    def get_residuals(self,
                      body_vel: Float[torch.Tensor, "B 6 1"],
                      body_vel_prev: Float[torch.Tensor, "B 6 1"],
                      lambda_n: Float[torch.Tensor, "B C 1"],
                      lambda_t: Float[torch.Tensor, "B 2C 1"],
                      lambda_j: Float[torch.Tensor, "5D 1"],
                      body_f: Float[torch.Tensor, "B 6 1"],
                      J_n: Float[torch.Tensor, "B C 6"],
                      J_t: Float[torch.Tensor, "B 2C 6"],
                      J_j_p: Float[torch.Tensor, "5D 6"],
                      J_j_c: Float[torch.Tensor, "5D 6"],
                      dt: float,
                      ) -> Float[torch.Tensor, "B 6 1"]:
        """
        Compute the residuals for the dynamics equations in a physics simulation.

        Args:
            body_vel: Current body velocities, shape [B, 6, 1], where B is the number of bodies.
            body_vel_prev: Previous body velocities, shape [B, 6, 1].
            lambda_n: Normal contact impulses, shape [B, C, 1], where C is the maximum number of contacts per body.
            lambda_t: Tangential friction impulses, shape [B, 2C, 1].
            lambda_j: Joint impulses, shape [5D, 1], where D is the number of joints.
            body_f: External forces on bodies, shape [B, 6, 1].
            J_n: Normal contact Jacobians, shape [B, C, 6].
            J_t: Tangential friction Jacobians, shape [B, 2C, 6].
            J_j_p: Joint Jacobians for parent bodies, shape [5D, 6].
            J_j_c: Joint Jacobians for child bodies, shape [5D, 6].
            dt: Time step (scalar).

        Returns:
            Residuals of the dynamics constraints, shape [B, 6, 1].
        """
        B = body_vel.shape[0]  # Body count
        D = self.joint_parent.shape[0]  # Joint count

        res = (
            torch.matmul(self.mass_matrix, (body_vel - body_vel_prev))  # M @ (v - v_prev)
            - torch.matmul(J_n.transpose(2, 1), lambda_n)  # J_n^T @ λ_n
            - torch.matmul(J_t.transpose(2, 1), lambda_t)  # J_t^T @ λ_t
            - body_f * dt  # f·dt
            - torch.matmul(self.mass_matrix, self.g_accel) * dt  # M @ g·dt
        ) # [B, 6, 1]

        joint_impulse = torch.zeros(B, 6, 1, device=self.device) # [B, 6, 1]

        # Reshape J_j_p and J_j_c for batch processing
        lambda_j_batch = lambda_j.view(D, 5, 1)  # [B, D, 1]

        impulse_p = torch.matmul(J_j_p.transpose(2, 1), lambda_j_batch)  # [B, 6, 1]
        impulse_c = torch.matmul(J_j_c.transpose(2, 1), lambda_j_batch)  # [B, 6, 1]

        # Scatter impulses to corresponding bodies
        joint_impulse.index_add_(0, self.joint_parent, impulse_p) # [B, 6, 1]
        joint_impulse.index_add_(0, self.joint_child, impulse_c) # [B, 6, 1]

        res = res - joint_impulse  # [B, 6, 1]

        return res

    def get_derivatives(self,
                        J_n: Float[torch.Tensor, "B C 6"],
                        J_t: Float[torch.Tensor, "B 2C 6"],
                        J_j_p: Float[torch.Tensor, "D 5 6"],
                        J_j_c: Float[torch.Tensor, "D 5 6"],
                        ) -> Tuple[Float[torch.Tensor, "B 6 6"],
                                   Float[torch.Tensor, "B 6 C"],
                                   Float[torch.Tensor, "B 6 2C"],
                                   Float[torch.Tensor, "B 6 5D"]]:
        """
        Compute the derivatives of the dynamics residuals with respect to velocities and impulses.

        Args:
            J_n: Normal contact Jacobians, shape [B, C, 6], where B is the number of bodies and C is the max contacts.
            J_t: Tangential friction Jacobians, shape [B, 2C, 6].
            J_j_p: Joint Jacobians for parent bodies, shape [D, 5, 6], where D is the number of joints.
            J_j_c: Joint Jacobians for child bodies, shape [D, 5, 6].

        Returns:
            Tuple containing:
                - ∂res/∂body_vel: Derivative w.r.t. body velocities, shape [B, 6, 6].
                - ∂res/∂lambda_n: Derivative w.r.t. normal impulses, shape [B, 6, C].
                - ∂res/∂lambda_t: Derivative w.r.t. tangential impulses, shape [B, 6, 2C].
                - ∂res/∂lambda_j: Derivative w.r.t. joint impulses, shape [B, 6, 5D].
        """
        B = J_n.shape[0]  # Body count
        D = self.joint_parent.shape[0]  # Joint count

        # ∂res/∂body_vel
        dres_dbody_vel = self.mass_matrix  # [B, 6, 6]

        # ∂res/∂lambda_n
        dres_dlambda_n = -J_n.transpose(1, 2)  # [B, 6, C]

        # ∂res/∂lambda_t
        dres_dlambda_t = -J_t.transpose(1, 2)  # [B, 6, 2C]

        # ∂res/∂lambda_j
        dres_dlambda_j = torch.zeros(B, 6, 5 * D, device=self.device)  # [B, 6, 5D]

        # Step 1: Reshape and transpose Jacobians
        J_j_p_reshaped = J_j_p.transpose(1, 2)  # [D, 6, 5]
        J_j_c_reshaped = J_j_c.transpose(1, 2)  # [D, 6, 5]

        # Step 2: Create block indices for column positions
        block_indices = (torch.arange(5, device=self.device).view(1, 1, 5) +
                         5 * torch.arange(D, device=self.device).view(D, 1, 1)).expand(D, 6, 5)  # [D, 6, 5]

        # Step 3: Create contribution tensors with scatter
        contrib_p = torch.zeros(D, 6, 5 * D, device=self.device).scatter_(2, block_indices, -J_j_p_reshaped) # [D, 6, 5D]
        contrib_c = torch.zeros(D, 6, 5 * D, device=self.device).scatter_(2, block_indices, -J_j_c_reshaped) # [D, 6, 5D]

        # Step 4: Accumulate contributions into dres_dlambda_j
        dres_dlambda_j.index_add_(0, self.joint_parent, contrib_p) # [B, 6, 5D]
        dres_dlambda_j.index_add_(0, self.joint_child, contrib_c) # [B, 6, 5D]

        return dres_dbody_vel, dres_dlambda_n, dres_dlambda_t, dres_dlambda_j

class ContactConstraint:
    def __init__(self,
        device: torch.device,
        fb_alpha: float = 0.3,
        fb_beta: float = 0.3,
        fb_epsilon: float = 1e-12,
        stabilization_factor: float = 0.2):
        self.device = device

        self.stabilization_factor = stabilization_factor
        self.fb = ScaledFisherBurmeister(alpha=fb_alpha, beta=fb_beta, epsilon=fb_epsilon)

    def get_penetration_depths(self,
                               body_trans: Float[torch.Tensor, "B 7 1"],
                               contact_points: Float[torch.Tensor, "B C 3 1"],
                               ground_points: Float[torch.Tensor, "B C 3 1"],
                               contact_normals: Float[torch.Tensor, "B C 3 1"]
                               ) -> Float[torch.Tensor, "B C 1"]:
        """
        Compute penetration depths for contact points in a physics simulation.

        Args:
            body_trans: Body transforms (position and quaternion), shape [B, 7, 1], where B is the number of bodies.
            contact_points: Contact points in local body frame, shape [B, C, 3, 1], where C is max contacts per body.
            ground_points: Corresponding ground points, shape [B, C, 3, 1].
            contact_normals: Contact normals in world frame, shape [B, C, 3, 1].

        Returns:
            Penetration depths for each contact, shape [B, C, 1].
        """
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
                                  body_trans: Float[torch.Tensor, "B 7 1"],
                                  contact_points: Float[torch.Tensor, "B C 3 1"],
                                  contact_normals: Float[torch.Tensor, "B C 3 1"],
                                  contact_mask: Bool[torch.Tensor, "B C"],
                                  ) -> Float[torch.Tensor, "B C 6"]:
        """
        Compute contact Jacobians for normal directions in a physics simulation.

        Args:
            body_trans: Body transforms, shape [B, 7, 1], where B is the number of bodies.
            contact_points: Contact points in local frame, shape [B, C, 3, 1], where C is max contacts per body.
            contact_normals: Contact normals in world frame, shape [B, C, 3, 1].
            contact_mask: Boolean mask for active contacts, shape [B, C].

        Returns:
            Contact Jacobians for normal directions, shape [B, C, 6].
        """
        B = body_trans.shape[0]  # Number of bodies
        C = contact_mask.shape[1]  # Max contacts per body

        # Transform local contact points to world frame
        q = body_trans[:, 3:].unsqueeze(1).expand(B, C, 4, 1)  # [B, C, 4, 1]
        r = rotate_vectors_batch(contact_points, q)  # [B, C, 3, 1]

        # Compute r × n for rotational component of the Jacobian
        r_cross_n = torch.cross(r, contact_normals, dim=2)  # [B, C, 3, 1]

        J = torch.cat([r_cross_n, contact_normals], dim=2).squeeze(-1)  # [B, C, 6]
        J[~contact_mask] = 0

        return J

    def get_residuals(self,
                      body_vel: Float[torch.Tensor, "B 6 1"],
                      body_vel_prev: Float[torch.Tensor, "B 6 1"],
                      lambda_n: Float[torch.Tensor, "B C 1"],
                      J_n: Float[torch.Tensor, "B C 6"],
                      penetration_depth: Float[torch.Tensor, "B C 1"],
                      contact_mask: Bool[torch.Tensor, "B C"],
                      contact_weight: Float[torch.Tensor, "B C"],
                      restitution: Float[torch.Tensor, "B 1"],
                      dt: float,
                      ) -> Float[torch.Tensor, "B C 1"]:
        """
        Compute residuals for contact constraints using a stabilized formulation.

        Args:
            body_vel: Current body velocities, shape [B, 6, 1], where B is the number of bodies.
            body_vel_prev: Previous body velocities, shape [B, 6, 1].
            lambda_n: Normal contact impulses, shape [B, C, 1], where C is max contacts per body.
            J_n: Normal contact Jacobians, shape [B, C, 6].
            penetration_depth: Penetration depths, shape [B, C, 1].
            contact_mask: Boolean mask for active contacts, shape [B, C].
            contact_weight: Contact weights, shape [B, C].
            restitution: Coefficients of restitution, shape [B, 1].
            dt: Time step (scalar).

        Returns:
            Residuals for contact constraints, shape [B, C, 1].
        """
        B = body_vel.shape[0] # Body count
        C = J_n.shape[1] # Maximum contact count per body

        active_mask = contact_mask.view(B, C, 1).float() # [B, C, 1]
        inactive_mask = 1 - active_mask # [B, C, 1]
        weight = contact_weight.view(B, C, 1).float() # [B, C, 1]

        v_n = torch.matmul(J_n, body_vel) # [B, C, 6] @ [B, 6, 1] -> [B, C, 1]
        v_n_prev = torch.matmul(J_n, body_vel_prev) # [B, C, 6] @ [B, 6, 1] -> [B, C, 1]
        e = restitution.view(B, 1, 1).expand(B, C, 1) # [B, C, 1]

        b_err = -(self.stabilization_factor / dt) * penetration_depth  # [B, C, 1]
        b_rest = e * v_n_prev # [B, C, 6] @ [B, C, 1] -> [B, C, 1]

        res_act = self.fb.evaluate(lambda_n, v_n + b_err + b_rest) # [B, C, 1]
        res_inact = -lambda_n # [B, C, 1]

        res = active_mask * res_act * weight + inactive_mask * res_inact # [B, C, 1]

        return res # [B, C, 1]

    def get_derivatives(self,
                        body_vel: Float[torch.Tensor, "B 6 1"],
                        body_vel_prev: Float[torch.Tensor, "B 6 1"],
                        lambda_n: Float[torch.Tensor, "B C 1"],
                        J_n: Float[torch.Tensor, "B C 6"],
                        penetration_depth: Float[torch.Tensor, "B C 1"],
                        contact_mask: Bool[torch.Tensor, "B C"],
                        contact_weight: Float[torch.Tensor, "B C"],
                        restitution: Float[torch.Tensor, "B 1"],
                        dt: float,
                        ) -> Tuple[Float[torch.Tensor, "B C 6"],
                                   Float[torch.Tensor, "B C C"]]:
        """
        Compute derivatives of contact residuals w.r.t. velocities and normal impulses.

        Args:
            body_vel: Current body velocities, shape [B, 6, 1], where B is the number of bodies.
            body_vel_prev: Previous body velocities, shape [B, 6, 1].
            lambda_n: Normal contact impulses, shape [B, C, 1], where C is max contacts per body.
            J_n: Normal contact Jacobians, shape [B, C, 6].
            penetration_depth: Penetration depths, shape [B, C, 1].
            contact_mask: Boolean mask for active contacts, shape [B, C].
            contact_weight: Contact weights, shape [B, C].
            restitution: Coefficients of restitution, shape [B, 1].
            dt: Time step (scalar).

        Returns:
            Tuple containing:
                - ∂res/∂body_vel: Derivative w.r.t. body velocities, shape [B, C, 6].
                - ∂res/∂lambda_n: Derivative w.r.t. normal impulses, shape [B, C, C].
        """
        B = body_vel.shape[0] # Body count
        C = J_n.shape[1] # Maximum contact count per body

        active_mask = contact_mask.view(B, C, 1).float() # [B, C, 1]
        inactive_mask = 1 - active_mask # [B, C, 1]
        weight = contact_weight.view(B, C, 1).float() # [B, C, 1]

        v_n = torch.matmul(J_n, body_vel) # [B, C, 6] @ [B, 6, 1] -> [B, C, 1]
        v_n_prev = torch.matmul(J_n, body_vel_prev) # [B, C, 6] @ [B, 6, 1] -> [B, C, 1]
        e = restitution.view(B, 1, 1).expand(B, C, 1) # [B, C, 1]

        b_err = -(self.stabilization_factor / dt) * penetration_depth  # [B, C, 1]
        b_rest = e * v_n_prev # [B, C, 6] @ [B, C, 1] -> [B, C, 1]

        da_act, db_act = self.fb.derivatives(lambda_n, v_n + b_err + b_rest) # [B, C, 1]
        da_n_inact = -torch.ones_like(lambda_n) # [B, C, 1]
        db_n_inact = torch.zeros_like(lambda_n) # [B, C, 1]

        da = da_act * weight * active_mask + da_n_inact * inactive_mask  # [B, C, 1]
        db = db_act * weight * active_mask + db_n_inact * inactive_mask  # [B, C, 1]

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

    @staticmethod
    def _compute_tangential_basis(
            contact_normals: Float[torch.Tensor, "B C 3 1"]
    ) -> Tuple[Float[torch.Tensor, "B C 3 1"],
    Float[torch.Tensor, "B C 3 1"]]:
        """
        Compute orthogonal tangent vectors for each contact normal.

        Args:
            contact_normals: Contact normals in world frame, shape [B, C, 3, 1], where B is bodies, C is max contacts.

        Returns:
            Tuple of tangent vectors (t1, t2), each shape [B, C, 3, 1], orthogonal to the normals.
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
                                     body_trans: Float[torch.Tensor, "B 7 1"],
                                     contact_points: Float[torch.Tensor, "B C 3 1"],
                                     contact_normals: Float[torch.Tensor, "B C 3 1"],
                                     contact_mask: Bool[torch.Tensor, "B C"]
                                     ) -> Float[torch.Tensor, "B 2C 6"]:
        """
        Compute tangential Jacobians for friction directions.

        Args:
            body_trans: Body transforms, shape [B, 7, 1], where B is the number of bodies.
            contact_points: Contact points in local frame, shape [B, C, 3, 1], where C is max contacts.
            contact_normals: Contact normals in world frame, shape [B, C, 3, 1].
            contact_mask: Boolean mask for active contacts, shape [B, C].

        Returns:
            Tangential Jacobians for friction directions, shape [B, 2C, 6].
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
        J_t = torch.cat((J_t1, J_t2), dim=1)  # [B, 2C, 6]

        return J_t

    def get_residuals(self,
                      body_vel: Float[torch.Tensor, "B 6 1"],
                      lambda_n: Float[torch.Tensor, "B C 1"],
                      lambda_t: Float[torch.Tensor, "B 2C 1"],
                      gamma: Float[torch.Tensor, "B C 1"],
                      J_t: Float[torch.Tensor, "B 2C 6"],
                      contact_mask: Bool[torch.Tensor, "B C"],
                      friction_coeff: Float[torch.Tensor, "B 1"]
                      ) -> Float[torch.Tensor, "B 3C 1"]:
        """
        Compute residuals for friction constraints in a physics simulation.

        Args:
            body_vel: Body velocities, shape [B, 6, 1], where B is the number of bodies.
            lambda_n: Normal contact impulses, shape [B, C, 1], where C is max contacts per body.
            lambda_t: Tangential friction impulses, shape [B, 2C, 1].
            gamma: Friction cone variables, shape [B, C, 1].
            J_t: Tangential Jacobians, shape [B, 2C, 6].
            contact_mask: Boolean mask for active contacts, shape [B, C].
            friction_coeff: Friction coefficients, shape [B, 1].

        Returns:
            Residuals for friction constraints, shape [B, 3C, 1].
        """
        B = body_vel.shape[0]
        C = lambda_n.shape[1]

        active_mask = contact_mask.view(B, C, 1).float() # [B, C, 1]
        inactive_mask = 1 - active_mask # [B, C, 1]

        # Compute tangential velocity
        v_t = torch.matmul(J_t, body_vel)  # [B, 2C, 1]
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

        res_fr = torch.cat([res_fr1, res_fr2], dim=1)  # [B, 2C, 1]

        res_frc_act = self.fb.evaluate(gamma, mu * lambda_n - lambda_t_norm) # [B, C, 1]
        res_frc_inact = - gamma # [B, C, 1]
        res_frc = res_frc_act * active_mask + res_frc_inact * inactive_mask # [B, C, 1]

        res = torch.cat([res_fr, res_frc], dim=1) # [B, 3C, 1]

        return res

    def get_derivatives(self,
                        body_vel: Float[torch.Tensor, "B 6 1"],
                        lambda_n: Float[torch.Tensor, "B C 1"],
                        lambda_t: Float[torch.Tensor, "B 2C 1"],
                        gamma: Float[torch.Tensor, "B C 1"],
                        J_t: Float[torch.Tensor, "B 2C 6"],
                        contact_mask: Bool[torch.Tensor, "B C"],
                        friction_coeff: Float[torch.Tensor, "B 1"],
                        ) -> Tuple[Float[torch.Tensor, "B 3C 6"],
                                   Float[torch.Tensor, "B 3C C"],
                                   Float[torch.Tensor, "B 3C 2C"],
                                   Float[torch.Tensor, "B 3C C"]]:
        """
        Compute derivatives of friction residuals w.r.t. various parameters.

        Args:
            body_vel: Body velocities, shape [B, 6, 1], where B is the number of bodies.
            lambda_n: Normal contact impulses, shape [B, C, 1], where C is max contacts per body.
            lambda_t: Tangential friction impulses, shape [B, 2C, 1].
            gamma: Friction cone variables, shape [B, C, 1].
            J_t: Tangential Jacobians, shape [B, 2C, 6].
            contact_mask: Boolean mask for active contacts, shape [B, C].
            friction_coeff: Friction coefficients, shape [B, 1].

        Returns:
            Tuple containing:
                - ∂res/∂body_vel: Derivative w.r.t. body velocities, shape [B, 3C, 6].
                - ∂res/∂lambda_n: Derivative w.r.t. normal impulses, shape [B, 3C, C].
                - ∂res/∂lambda_t: Derivative w.r.t. tangential impulses, shape [B, 3C, 2C].
                - ∂res/∂gamma: Derivative w.r.t. friction cone variables, shape [B, 3C, C].
        """
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
        dres_fr_dbody_vel = torch.cat([dres_fr1_dbody_vel, dres_fr2_dbody_vel], dim=1)  # [B, 2C, 6]

        # ∂res_fr / ∂lambda_n
        dres_fr_dlambda_n = torch.zeros((B, 2 * C, C), device=self.device) # [B, 2C, C]

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
        dres_fr1_dlambda_t = torch.cat([dres_fr1_dlambda_t1, dres_fr1_dlambda_t2], dim=2)  # [B, C, 2C]

        # Combine derivatives for res_fr2
        dres_fr2_dlambda_t = torch.cat([dres_fr2_dlambda_t1, dres_fr2_dlambda_t2], dim=2)  # [B, C, 2C]

        # Combine for complete ∂res_fr / ∂lambda_t
        dres_fr_dlambda_t = torch.cat([dres_fr1_dlambda_t, dres_fr2_dlambda_t], dim=1)  # [B, 2C, 2C]

        # ∂res_fr / ∂gamma
        dres_fr1_dgamma = (lambda_t1 / (lambda_t_norm + self.eps)) * active_mask  # [B, C, 1]
        dres_fr1_dgamma = torch.diag_embed(dres_fr1_dgamma.squeeze(-1))  # [B, C, C]

        dres_fr2_dgamma = (lambda_t2 / (lambda_t_norm + self.eps)) * active_mask  # [B, C, 1]
        dres_fr2_dgamma = torch.diag_embed(dres_fr2_dgamma.squeeze(-1))  # [B, C, C]

        dres_fr_dgamma = torch.cat([dres_fr1_dgamma, dres_fr2_dgamma], dim=1)  # [B, 2C, C]

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

        dres_frc_dlambda_t = torch.cat([dres_frc_dlambda_t1, dres_frc_dlambda_t2], dim=2)  # [B, C, 2C]

        # ∂res_frc / ∂gamma
        dres_frc_dgamma_inact = -torch.ones_like(gamma)  # [B, C, 1]
        dres_frc_dgamma = dres_frc_dgamma_act * active_mask + dres_frc_dgamma_inact * inactive_mask  # [B, C, 1]
        dres_frc_dgamma = torch.diag_embed(dres_frc_dgamma.squeeze(-1))  # [B, C, C]

        # Combine all derivatives
        # ∂res / ∂body_vel
        dres_dbody_vel = torch.cat([dres_fr_dbody_vel, dres_frc_dbody_vel], dim=1)  # [B, 3C, 6]

        # ∂res / ∂lambda_n
        dres_dlambda_n = torch.cat([dres_fr_dlambda_n, dres_frc_dlambda_n], dim=1)  # [B, 3C, C]

        # ∂res / ∂lambda_t
        dres_dlambda_t = torch.cat([dres_fr_dlambda_t, dres_frc_dlambda_t], dim=1)  # [B, 3C, 2C]

        # ∂res / ∂gamma
        dres_dgamma = torch.cat([dres_fr_dgamma, dres_frc_dgamma], dim=1)  # [B, 3C, C]

        return dres_dbody_vel, dres_dlambda_n, dres_dlambda_t, dres_dgamma
    
class RevoluteConstraint:
    def __init__(self, model: Model):
        self.joint_parent = model.joint_parent
        self.joint_child = model.joint_child
        
        self.joint_trans_parent = model.joint_X_p
        self.joint_trans_child = model.joint_X_c
        
        self.device = model.device
        
        self.stabilization_factor = 0.2
        self.eps = 1e-10
        self.weight = 1
        
    
    def _get_joint_frames(self,
                          body_trans: Float[torch.Tensor, "B 7 1"]
                          ) -> Tuple[Float[torch.Tensor, "D 7 1"],
                                     Float[torch.Tensor, "D 7 1"],
                                     Float[torch.Tensor, "D 3 1"],
                                     Float[torch.Tensor, "D 3 1"]]:
        """
        Compute joint frames and relative vectors for parent and child bodies.

        Args:
            body_trans: Body transforms, shape [B, 7, 1], where B is the number of bodies.

        Returns:
            Tuple containing:
                - X_p: Parent joint frames, shape [D, 7, 1], where D is the number of joints.
                - X_c: Child joint frames, shape [D, 7, 1].
                - r_p: Relative vectors for parent, shape [D, 3, 1].
                - r_c: Relative vectors for child, shape [D, 3, 1].
        """
        trans_parent = body_trans[self.joint_parent]
        trans_child = body_trans[self.joint_child]

        # Joint frame computed from the parent and child bodies
        X_p = transform_multiply_batch(trans_parent, self.joint_trans_parent)  # [D, 7, 1]
        X_c = transform_multiply_batch(trans_child, self.joint_trans_child) # [D, 7, 1]

        # Get the relative vector from the body frame to joint frame in world coordinates
        r_p = X_p[:, :3] - trans_parent[:, :3] # [D, 3, 1]
        r_c = X_c[:, :3] - trans_child[:, :3] # [D, 3, 1]

        return X_p, X_c, r_p, r_c

    def _get_translational_errors(self,
                                  X_p: Float[torch.Tensor, "D 7 1"],
                                  X_c: Float[torch.Tensor, "D 7 1"]
                                  ) -> Tuple[Float[torch.Tensor, "D 1"],
                                             Float[torch.Tensor, "D 1"],
                                             Float[torch.Tensor, "D 1"]]:
        """
        Compute translational errors between parent and child joint frames.

        Args:
            X_p: Parent joint frames, shape [D, 7, 1], where D is the number of joints.
            X_c: Child joint frames, shape [D, 7, 1].

        Returns:
            Tuple of translational errors (x, y, z), each shape [D, 1].
        """
        err_x = X_c[:, 0] - X_p[:, 0] # [D, 1]
        err_y = X_c[:, 1] - X_p[:, 1] # [D, 1]
        err_z = X_c[:, 2] - X_p[:, 2] # [D, 1]

        return err_x, err_y, err_z

    def _get_rotational_errors(self,
                               X_p: Float[torch.Tensor, "D 7 1"],
                               X_c: Float[torch.Tensor, "D 7 1"]
                               ) -> Tuple[Float[torch.Tensor, "D 1"],
                                          Float[torch.Tensor, "D 1"]]:
        """
        Compute rotational errors for revolute joints.

        Args:
            X_p: Parent joint frames, shape [D, 7, 1], where D is the number of joints.
            X_c: Child joint frames, shape [D, 7, 1].

        Returns:
            Tuple of rotational errors (x, y), each shape [D, 1].
        """
        D = X_p.shape[0]

        # Create unit vectors for x, y, and z axes
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(D, 1)  # [D, 3]
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(D, 1)  # [D, 3]
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(D, 1)  # [D, 3]

        # Rotate unit vectors to child and parent frames
        x_axis_c = rotate_vectors_batch(x_axis.unsqueeze(-1), X_c[:, 3:]) # [D, 3, 1]
        y_axis_c = rotate_vectors_batch(y_axis.unsqueeze(-1), X_c[:, 3:]) # [D, 3, 1]
        z_axis_p = rotate_vectors_batch(z_axis.unsqueeze(-1), X_p[:, 3:]) # [D, 3, 1]

        # Rotational constraints: x and y of child perpendicular to z of parent
        err_x = torch.matmul(x_axis_c.transpose(1, 2), z_axis_p).squeeze(-1)  # [D, 1]
        err_y = torch.matmul(y_axis_c.transpose(1, 2), z_axis_p).squeeze(-1)  # [D, 1]

        return err_x, err_y

    def get_errors(self,
                   body_trans: Float[torch.Tensor, "B 7 1"]
                   ) -> Float[torch.Tensor, "5D 1"]:
        """
        Compute position-level errors for revolute joints.

        Args:
            body_trans: Body transforms, shape [B, 7, 1], where B is the number of bodies.

        Returns:
            Errors for each joint constraint, shape [5D, 1], where D is the number of joints.
        """
        X_p, X_c, r_p, r_c = self._get_joint_frames(body_trans)  # [D, 7, 1], [D, 7, 1]

        err_tx, err_ty, err_tz = self._get_translational_errors(X_p, X_c)  # [D, 1], [D, 1], [D, 1]

        err_rx, err_ry = self._get_rotational_errors(X_p, X_c)  # [D, 1], [D, 1]

        errors = torch.stack([err_tx, err_ty, err_tz, err_rx, err_ry], dim=1)  # [D, 5, 1]

        return errors

    def _get_translational_jacobians(self,
                                     r_p: Float[torch.Tensor, "D 3 1"],
                                     r_c: Float[torch.Tensor, "D 3 1"]
                                     ) -> Tuple[Float[torch.Tensor, "D 3 6"],
                                                Float[torch.Tensor, "D 3 6"]]:
        """
        Compute translational Jacobians for revolute joints.

        Args:
            r_p: Relative vectors for parent, shape [D, 3, 1], where D is the number of joints.
            r_c: Relative vectors for child, shape [D, 3, 1].

        Returns:
            Tuple of translational Jacobians for parent and child, each shape [D, 3, 6].
        """
        D = r_p.shape[0]  # Number of joints

        # Create the 3x3 identity matrix E_3 and repeat for each joint
        E_3 = torch.eye(3, device=self.device).unsqueeze(0).repeat(D, 1, 1)  # [D, 3, 3]

        # Compute skew-symmetric matrices for r_p and r_c
        r_skew_p = skew_symmetric_matrix_batch(r_p)  # [D, 3, 3]
        r_skew_c = skew_symmetric_matrix_batch(r_c)  # [D, 3, 3]

        J_p = torch.cat([r_skew_p, -E_3], dim=2)  # Shape [D, 3, 6]
        J_c = torch.cat([-r_skew_c, E_3], dim=2)  # Shape [D, 3, 6]

        return J_p, J_c

    def _get_rotational_jacobians(self,
                                  X_p: Float[torch.Tensor, "D 7 1"],
                                  X_c: Float[torch.Tensor, "D 7 1"],
                                  ) -> Tuple[Float[torch.Tensor, "D 2 6"],
                                             Float[torch.Tensor, "D 2 6"]]:
        """
        Compute rotational Jacobians for revolute joints.

        Args:
            X_p: Parent joint frames, shape [D, 7, 1], where D is the number of joints.
            X_c: Child joint frames, shape [D, 7, 1].

        Returns:
            Tuple of rotational Jacobians for parent and child, each shape [D, 2, 6].
        """
        D = X_p.shape[0]

        # Create unit vectors for x, y, and z axes
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(D, 1)  # [D, 3]
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(D, 1)  # [D, 3]
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(D, 1)  # [D, 3]

        # Rotate unit vectors to child and parent frames
        x_axis_c = rotate_vectors_batch(x_axis.unsqueeze(-1), X_c[:, 3:]) # [D, 3, 1]
        y_axis_c = rotate_vectors_batch(y_axis.unsqueeze(-1), X_c[:, 3:]) # [D, 3, 1]
        z_axis_p = rotate_vectors_batch(z_axis.unsqueeze(-1), X_p[:, 3:]) # [D, 3, 1]

        x_c_cross_z_p = torch.linalg.cross(x_axis_c, z_axis_p, dim=1).squeeze(-1) # [D, 3]
        y_c_cross_z_p = torch.linalg.cross(y_axis_c, z_axis_p, dim=1).squeeze(-1) # [D, 3]

        zeros = torch.zeros((D, 3), device=self.device, dtype=x_c_cross_z_p.dtype) # [D, 3]

        Jx_p = torch.cat([-x_c_cross_z_p, zeros], dim=1) # [D, 6]
        Jy_p = torch.cat([-y_c_cross_z_p, zeros], dim=1) # [D, 6]

        Jx_c = torch.cat([x_c_cross_z_p, zeros], dim=1) # [D, 6]
        Jy_c = torch.cat([y_c_cross_z_p, zeros], dim=1) # [D, 6]

        J_p = torch.stack([Jx_p, Jy_p], dim=1) # [D, 2, 6]
        J_c = torch.stack([Jx_c, Jy_c], dim=1) # [D, 2, 6]

        return J_p, J_c

    def compute_jacobians(self,
                          body_trans: Float[torch.Tensor, "B 7 1"]
                          ) -> Tuple[Float[torch.Tensor, "5D 6"],
                                     Float[torch.Tensor, "5D 6"]]:
        """
        Compute Jacobians for revolute joint constraints.

        Args:
            body_trans: Body transforms, shape [B, 7, 1], where B is the number of bodies.

        Returns:
            Tuple of Jacobians for parent and child bodies, each shape [5D, 6], where D is the number of joints.
        """
        D = self.joint_parent.shape[0]
        X_p, X_c, r_p, r_c = self._get_joint_frames(body_trans) # [D, 7, 1], [D, 7, 1], [D, 3, 1], [D, 3, 1]

        # Get translational Jacobians (3 constraints per joint)
        Jt_p, Jt_c = self._get_translational_jacobians(r_p, r_c) # [D, 3, 6], [D, 3, 6]

        # Get rotational Jacobians (2 constraints per joint)
        Jr_p, Jr_c = self._get_rotational_jacobians(X_p, X_c) # [D, 2, 6], [D, 2, 6]

        # Combine translational and rotational Jacobians
        J_p = torch.cat([Jt_p, Jr_p], dim=1)  # [D, 5, 6]
        J_c = torch.cat([Jt_c, Jr_c], dim=1)  # [D, 5, 6]

        return J_p, J_c

    def get_residuals(self,
                      body_vel: Float[torch.Tensor, "B 6 1"],
                      body_trans: Float[torch.Tensor, "B 7 1"],
                      J_j_p: Float[torch.Tensor, "5D 6"],
                      J_j_c: Float[torch.Tensor, "5D 6"],
                      dt: float
                      ) -> Float[torch.Tensor, "5D 1"]:
        """
        Compute residuals for revolute joint constraints with stabilization.

        Args:
            body_vel: Body velocities, shape [B, 6, 1], where B is the number of bodies.
            body_trans: Body transforms, shape [B, 7, 1].
            J_j_p: Joint Jacobians for parent bodies, shape [5D, 6].
            J_j_c: Joint Jacobians for child bodies, shape [5D, 6].
            dt: Time step (scalar).

        Returns:
            Residuals for joint constraints, shape [5D, 1].
        """
        D = self.joint_parent.shape[0]

        # Get position-level errors
        errors = self.get_errors(body_trans) # [D, 5, 1]

        body_vel_p = body_vel[self.joint_parent]
        body_vel_c = body_vel[self.joint_child]

        # Compute velocity-level residuals with Baumgarte stabilization
        v_j_p = torch.matmul(J_j_p, body_vel_p)  # [D, 5, 6] @ [D, 6, 1] -> [D, 5, 1]
        v_j_c = torch.matmul(J_j_c, body_vel_c)  # [D, 5, 6] @ [D, 6, 1] -> [D, 5, 1]

        bias = (self.stabilization_factor / dt) * errors  # [D, 5, 1]

        res = v_j_p + v_j_c + bias  # [D, 5, 1]

        return res.view(5 * D, 1)  # [5D, 1]

    def get_derivatives(self,
                        body_vel: Float[torch.Tensor, "B 6 1"],
                        J_j_p: Float[torch.Tensor, "5D 6"],
                        J_j_c: Float[torch.Tensor, "5D 6"],
                        ) -> Float[torch.Tensor, "5D 6B"]:
        """
        Compute derivatives of joint residuals w.r.t. body velocities.

        Args:
            body_vel: Body velocities, shape [B, 6, 1], where B is the number of bodies.
            J_j_p: Joint Jacobians for parent bodies, shape [5D, 6], where D is the number of joints.
            J_j_c: Joint Jacobians for child bodies, shape [5D, 6].

        Returns:
            Derivative ∂res/∂body_vel, shape [5D, 6B].
        """
        D = self.joint_parent.shape[0]
        B = body_vel.shape[0]

        # Initialize dense matrix
        dres_joint_dbody_vel = torch.zeros((5 * D, 6 * B), device=self.device)

        # Vectorized assignment
        row_indices = torch.arange(5 * D, device=self.device)  # [5D]
        parent_col_base = 6 * self.joint_parent # [D]
        child_col_base = 6 * self.joint_child  # [D]
        col_offsets = torch.arange(6, device=self.device)  # [6]

        # Expand indices for broadcasting
        parent_cols = (parent_col_base.repeat_interleave(5)[:, None] + col_offsets).flatten()  # [5⋅D⋅6]
        child_cols = (child_col_base.repeat_interleave(5)[:, None] + col_offsets).flatten()  # [5⋅D⋅6]
        rows = row_indices.repeat_interleave(6)  # [5⋅D⋅6]

        # Assign values directly
        dres_joint_dbody_vel[rows, parent_cols] = J_j_p.flatten() # [5D, 6B]
        dres_joint_dbody_vel[rows, child_cols] = J_j_c.flatten() # [5D, 6B]

        return dres_joint_dbody_vel
