from typing import Tuple

import torch

from pbd_torch.constraints import revolute_constraint_values
from pbd_torch.integrator import integrate_quat_exact_batch
from pbd_torch.integrator import SemiImplicitEulerIntegrator
from pbd_torch.model import Control
from pbd_torch.model import Model
from pbd_torch.model import State
from pbd_torch.ncp import ScaledFisherBurmeister
from pbd_torch.transform import transform_points_batch
from pbd_torch.transform import rotate_vectors_batch
from jaxtyping import Float, Bool
from pbd_torch.constraints import ground_penetration
from pbd_torch.constraints import compute_contact_jacobians
from pbd_torch.constraints import compute_tangential_jacobians

BETA = 0.2  # Damping coefficient for Baumgarte stabilization

# B ... body count
# C ... maximum number contacts per body
# D ... maximum number of joints per body

def compute_dynamics_residual(
    mass_matrix: Float[torch.Tensor, "body_count 6 6"],
    body_vel: Float[torch.Tensor, "body_count 6 1"],
    body_vel_prev: Float[torch.Tensor, "body_count 6 1"],
    body_f: Float[torch.Tensor, "body_count 6 1"],
    lambda_n: Float[torch.Tensor, "body_count max_contacts 1"],
    lambda_t: Float[torch.Tensor, "body_count max_contacts 2 1"],
    J_n: Float[torch.Tensor, "body_count max_contacts 6"],
    J_t: Float[torch.Tensor, "body_count max_contacts 2 6"],
    g_accel: Float[torch.Tensor, "3 1"],
    dt: float,
) -> Float[torch.Tensor, "body_count 6 1"]:
    """Compute the dynamics residual for Newton's method.

    This function calculates the residual of the dynamics equation:
    M @ (v - v_prev) - J_n^T @ λ_n - J_t^T @ λ_t - f·dt - M @ g·dt = 0

    Args:
        mass_matrix: Mass matrix for each body
        body_vel: Current body velocities
        body_vel_prev: Previous body velocities
        body_f: External forces acting on each body
        lambda_n: Normal contact forces
        lambda_t: Tangential contact forces
        J_n: Normal contact Jacobians
        J_t: Tangential contact Jacobians
        g_accel: Gravity acceleration vector
        dt: Time step size

    Returns:
        Dynamics residual
    """
    return (
        torch.bmm(mass_matrix, (body_vel - body_vel_prev))  # M @ (v - v_prev)
        - torch.bmm(J_n.transpose(1, 2), lambda_n)  # J_n^T @ λ_n
        - torch.matmul(J_t.transpose(2, 3), lambda_t).sum(dim=1)  # J_t^T @ λ_t
        - body_f * dt  # f·dt
        - torch.bmm(mass_matrix, g_accel.expand(body_vel.shape[0], 6, 1)) * dt  # M @ g·dt
    )


class NonSmoothNewtonEngine:
    def __init__(
        self, iterations: int = 10, device: torch.device = torch.device("cpu")
    ):
        self.device = device
        self.iterations = iterations
        self.integrator = SemiImplicitEulerIntegrator(
            use_local_omega=False, device=device
        )
        self.fb_contact = ScaledFisherBurmeister(alpha=0.3, beta=0.3, epsilon=1e-6)
        self.fb_friction = ScaledFisherBurmeister(alpha=0.3, beta=0.3, epsilon=1e-6)
        self.reg = 1e-6  # Regularization for Jacobian
        self.eps = 1e-12  # Regularization for square root and division

    def compute_residuals(
        self,
        model: Model,
        state_in: State,
        body_vel: Float[torch.Tensor, "body_count 6 1"],
        lambda_n: Float[torch.Tensor, "body_count max_contacts 1"],
        lambda_t: Float[torch.Tensor, "body_count max_contacts 2 1"],
        gamma: Float[torch.Tensor, "body_count max_contacts 1"],
        J_n: Float[torch.Tensor, "body_count max_contacts 6"],
        J_t: Float[torch.Tensor, "body_count max_contacts 2 6"],
        v_n_prev: Float[torch.Tensor, "body_count max_contacts 1"],
        dt: float,
    ) -> Tuple[
        Float[torch.Tensor, "body_count 6 1"],
        Float[torch.Tensor, "body_count max_contacts 1"],
        Float[torch.Tensor, "body_count max_contacts 1"],
        Float[torch.Tensor, "body_count max_contacts 1"],
        Float[torch.Tensor, "body_count max_contacts 1"]
    ]:
        """Compute the residuals for the non-smooth Newton method.

        This function calculates all residuals for the contact dynamics problem:
        - dynamics residual
        - normal contact complementarity residual
        - friction residuals
        - friction complementarity residual

        Args:
            model: Physical model containing mass matrix and friction coefficients
            state_in: Input state with body positions, velocities, and contacts
            body_vel: Current body velocities
            lambda_n: Normal contact forces
            lambda_t: Tangential contact forces
            gamma: Auxiliary variables for friction cone constraint
            dt: Time step
            J_n: Normal contact Jacobians
            J_t: Tangential contact Jacobians
            v_n_prev: Previous normal velocities for restitution

        Returns:
            Tuple of residuals (res_d, res_n, res_fr1, res_fr2, res_frc)
        """
        B = model.body_count  # Body count
        C = state_in.contact_points.shape[1]  # Contact count

        # DEBUG
        # if state_in.time > 0.2:
        #     constraint = revolute_constraint_values(state_in.body_q, model.joint_parent, model.joint_child, model.joint_X_p, model.joint_X_c, 3)

        # Compute the active and inactive contact masks
        active_mask = state_in.contact_mask.unsqueeze(-1).float()  # [B, C, 1]
        inactive_mask = 1 - active_mask  # [B, C, 1]

        # Compute the relative velocity at the contact points in each basis direction
        v_n = torch.bmm(J_n, body_vel)  # [B, C, 1]
        v_t = torch.einsum("nckj,njl->nckl", J_t, body_vel)  # [B, C, 2, 1]

        # Compute friction impulse norm
        lambda_t_norm = torch.sqrt(
            torch.sum(lambda_t**2, dim=2, keepdim=True) + self.eps
        )  # [B, C, 1, 1]

        # Extend the friction and restitution coefficients to the contact points
        e = model.restitution.unsqueeze(1).expand(B, C, 1)  # [B, C, 1]
        mu = model.dynamic_friction.unsqueeze(1).expand(B, C, 1)  # [B, C, 1]

        # Compute the penetration depth
        penetration_depth = ground_penetration(
            state_in.body_q,
            state_in.contact_points,
            state_in.contact_points_ground,
            state_in.contact_normals,
        )  # [B, C, 1]

        # ------------------ Dynamics residual ------------------
        # Dynamics residual: M @ Δbody_qd - J_n^T @ lambda_n - J_t1 @ lambda_t1 - J_t2 @ lambda_t2 - f * dt - M @ g = 0
        res_d = compute_dynamics_residual(
            model.mass_matrix,
            body_vel,
            state_in.body_qd,
            state_in.body_f,
            lambda_n,
            lambda_t,
            J_n,
            J_t,
            model.gravity,
            dt,
        ) # [B, 6, 1]

        # ------------------ Normal contact residual ------------------
        # Compute the error and restitution biases
        b_err = -(BETA / dt) * penetration_depth  # [B, C, 1]
        b_rest = e * v_n_prev  # [B, C, 1]
        b_n = b_err + b_rest  # [B, C, 1]

        # Normal contact residual is complementarity condition approximated by
        # Fisher-Burmeister function
        res_n_active = self.fb_contact.evaluate(lambda_n, v_n + b_n)  # [B, C, 1]
        res_n_inactive = -lambda_n  # [B, C, 1]
        res_n = res_n_active * active_mask + res_n_inactive * inactive_mask  # [B, C, 1]

        # ------------------ Friction contact residuals ------------------
        res_fr_active = v_t + gamma.unsqueeze(2) * lambda_t / (
            lambda_t_norm + self.eps
        )  # [B, C, 2, 1]
        res_fr_inactive = -lambda_t  # [B, C, 1]

        res_fr = res_fr_active * active_mask.unsqueeze(
            2
        ) + res_fr_inactive * inactive_mask.unsqueeze(
            2
        )  # [B, C, 1]

        # Compute the friction complementarity condition
        res_frc_active = self.fb_friction.evaluate(
            gamma, mu * lambda_n - lambda_t_norm.squeeze(3)
        )  # [B, C, 1]
        res_frc_inactive = -gamma  # [B, C, 1]
        res_frc = (
            res_frc_active * active_mask + res_frc_inactive * inactive_mask
        )  # [B, C, 1]

        return res_d, res_n, res_fr[:, :, 0, :], res_fr[:, :, 1, :], res_frc

    def compute_jacobian(
        self,
        model: Model,
        state_in: State,
        body_vel: Float[torch.Tensor, "body_count 6 1"],
        lambda_n: Float[torch.Tensor, "body_count max_contacts 1"],
        lambda_t: Float[torch.Tensor, "body_count max_contacts 2 1"],
        gamma: Float[torch.Tensor, "body_count max_contacts 1"],
        J_n: Float[torch.Tensor, "body_count max_contacts 6"],
        J_t: Float[torch.Tensor, "body_count max_contacts 2 6"],
        v_n_prev: Float[torch.Tensor, "body_count max_contacts 1"],
        dt: float,
    ) -> Float[torch.Tensor, "body_count total_dim total_dim"]:
        """Compute the Jacobian matrix for the non-smooth Newton method.

        This function calculates the analytical Jacobian of all residuals
        with respect to all variables (body_qd, lambda_n, lambda_t, gamma).

        Args:
            model: Physical model containing mass and friction parameters
            state_in: Input state with body position, contacts, etc.
            body_vel: Current body velocities
            lambda_n: Normal contact forces
            lambda_t: Tangential contact forces
            gamma: Auxiliary variables for friction cone constraint
            J_n: Normal contact Jacobians
            J_t: Tangential contact Jacobians
            v_n_prev: Previous normal velocities for restitution
            dt: Time step

        Returns:
            Full Jacobian matrix for Newton iteration
        """

        def diag(v: torch.Tensor):
            return torch.diag_embed(v.squeeze(-1))

        B = model.body_count  # Body count
        C = state_in.contact_points.shape[1]  # Maximum number of contacts
        total_dim = 6 + 4 * C
        J_F = torch.zeros((B, total_dim, total_dim), device=self.device)

        # Compute the active and inactive contact masks
        active_mask = state_in.contact_mask.unsqueeze(-1).float()  # [B, C, 1]
        inactive_mask = 1 - active_mask  # [B, C, 1]

        # Compute the relative velocity at the contact points in each basis direction
        v_n = torch.bmm(J_n, body_vel)  # [B, C, 1]

        lambda_t1 = lambda_t[:, :, 0, :]  # [B, C, 1]
        lambda_t2 = lambda_t[:, :, 1, :]  # [B, C, 1]
        lambda_t_norm = torch.sqrt(lambda_t1**2 + lambda_t2**2 + self.eps)  # [B, C, 1]

        e = model.restitution.unsqueeze(1).expand(B, C, 1)  # [B, C, 1]
        mu = model.dynamic_friction.unsqueeze(1).expand(B, C, 1)  # [B, C, 1]

        # Compute the penetration depth
        penetration_depth = ground_penetration(
            state_in.body_q,
            state_in.contact_points,
            state_in.contact_points_ground,
            state_in.contact_normals,
        )  # [B, C, 1]

        b_err = -(BETA / dt) * penetration_depth  # [B, C, 1]
        b_rest = e * v_n_prev  # [B, C, 1]
        b_n = b_err + b_rest  # [B, C, 1]

        # Partial derivatives of the contact normal Fisher-Burmeister function with respect to its arguments
        da_n_active, db_n_active = self.fb_contact.derivatives(
            lambda_n, v_n + b_n
        )  # [B, C, 1]

        # ∂res_n / ∂lambda_n
        da_n_inactive = -torch.ones_like(lambda_n)  # [B, C, 1]
        da_n = da_n_active * active_mask + da_n_inactive * inactive_mask  # [B, C, 1]

        # ∂res_n / ∂(v_n + b_n)
        db_n_inactive = torch.zeros_like(lambda_n)  # [B, C, 1]
        db_n = db_n_active * active_mask + db_n_inactive * inactive_mask  # [B, C, 1]

        # Partial derivatives of the friction Fisher-Burmeister function with respect to its arguments
        da_frc_active, db_frc_active = self.fb_friction.derivatives(
            gamma, mu * lambda_n - lambda_t_norm
        )  # [B, C, 1]

        # ∂res_frc / ∂gamma
        da_frc_inactive = -torch.ones_like(gamma)  # [B, C, 1]
        da_frc = (
            da_frc_active * active_mask + da_frc_inactive * inactive_mask
        )  # [B, C, 1]

        # ∂res_frc / ∂(mu * lambda_n - f_t_norm)
        db_frc_inactive = torch.zeros_like(gamma)  # [B, C, 1]
        db_frc = (
            db_frc_active * active_mask + db_frc_inactive * inactive_mask
        )  # [B, C, 1]

        # Compute the friction force norm derivatives
        lambda_t1_normalized = lambda_t1 / (lambda_t_norm + self.eps)  # [B, C, 1]
        lambda_t2_normalized = lambda_t2 / (lambda_t_norm + self.eps)  # [B, C, 1]

        df_t1_fr1_active = (
            gamma * (lambda_t2**2 + self.eps) / (lambda_t_norm**3 + self.eps)
        )
        df_t2_fr2_active = (
            gamma * (lambda_t1**2 + self.eps) / (lambda_t_norm**3 + self.eps)
        )

        df_t1_fr1_inactive = -torch.ones_like(lambda_t1)  # [B, C, 1]
        df_t1_fr1 = (
            df_t1_fr1_active * active_mask + df_t1_fr1_inactive * inactive_mask
        )  # [B, C, 1]

        df_t2_fr2_inactive = -torch.ones_like(lambda_t2)  # [B, C, 1]
        df_t2_fr2 = (
            df_t2_fr2_active * active_mask + df_t2_fr2_inactive * inactive_mask
        )  # [B, C, 1]

        df_t1_fr2 = (
            -gamma * lambda_t1 * lambda_t2 / (lambda_t_norm**3 + self.eps)
        )  # [B, C, 1]
        df_t2_fr1 = (
            -gamma * lambda_t1 * lambda_t2 / (lambda_t_norm**3 + self.eps)
        )  # [B, C, 1]

        # Jacobian blocks

        # ------------------ Dynamics residual derivatives ------------------
        J_F[:, :6, :6] = model.mass_matrix  # ∂res_d/∂body_qd
        J_F[:, :6, 6 : 6 + C] = -J_n.transpose(1, 2)  # ∂res_d/∂lambda_n
        J_F[:, :6, 6 + C : 6 + 2 * C] = -J_t[:, :, 0, :].transpose(1, 2)  # ∂res_d/∂f_t1
        J_F[:, :6, 6 + 2 * C : 6 + 3 * C] = -J_t[:, :, 1, :].transpose(
            1, 2
        )  # ∂res_d/∂f_t2

        # ------------------ Normal contact residual derivatives ------------------
        J_F[:, 6 : 6 + C, :6] = db_n * J_n  # ∂res_n/∂body_qd
        J_F[:, 6 : 6 + C, 6 : 6 + C] = diag(da_n)  # ∂res_n/∂lambda_n

        # ------------------ Friction contact residuals derivatives ------------------
        # ∂res_fr1/∂body_qd
        J_F[:, 6 + C : 6 + 2 * C, :6] = J_t[:, :, 0, :] * active_mask.expand(-1, -1, 6)
        # ∂res_fr1/∂f_t1
        J_F[:, 6 + C : 6 + 2 * C, 6 + C : 6 + 2 * C] = diag(df_t1_fr1)
        # ∂res_fr1/∂f_t2
        J_F[:, 6 + C : 6 + 2 * C, 6 + 2 * C : 6 + 3 * C] = diag(df_t1_fr2)
        # ∂res_fr1/∂gamma
        J_F[:, 6 + C : 6 + 2 * C, 6 + 3 * C : 6 + 4 * C] = diag(lambda_t1_normalized)

        # res_friction2 derivatives
        # ∂res_fr2/∂body_qd
        J_F[:, 6 + 2 * C : 6 + 3 * C, :6] = J_t[:, :, 1, :] * active_mask.expand(
            -1, -1, 6
        )
        # ∂res_fr2/∂f_t1
        J_F[:, 6 + 2 * C : 6 + 3 * C, 6 + C : 6 + 2 * C] = diag(df_t2_fr1)
        # ∂res_fr2/∂f_t2
        J_F[:, 6 + 2 * C : 6 + 3 * C, 6 + 2 * C : 6 + 3 * C] = diag(df_t2_fr2)
        # ∂res_fr2/∂gamma
        J_F[:, 6 + 2 * C : 6 + 3 * C, 6 + 3 * C : 6 + 4 * C] = diag(
            lambda_t2_normalized
        )

        # res_friction_complementarity derivatives
        J_F[:, 6 + 3 * C : 6 + 4 * C, 6 : 6 + C] = diag(
            db_frc * mu
        )  # ∂res_frc/∂lambda_n
        # ∂res_frc/∂f_t1
        J_F[:, 6 + 3 * C : 6 + 4 * C, 6 + C : 6 + 2 * C] = diag(
            -db_frc * lambda_t1_normalized
        )
        # ∂res_frc/∂f_t2
        J_F[:, 6 + 3 * C : 6 + 4 * C, 6 + 2 * C : 6 + 3 * C] = diag(
            -db_frc * lambda_t2_normalized
        )
        # ∂res_frc/∂gamma
        J_F[:, 6 + 3 * C : 6 + 4 * C, 6 + 3 * C : 6 + 4 * C] = diag(da_frc)

        return J_F

    def check_jacobian(
        self,
        model: Model,
        state_in: State,
        J_F: Float[torch.Tensor, "body_count total_dim total_dim"],
        body_vel: Float[torch.Tensor, "body_count 6 1"],
        lambda_n: Float[torch.Tensor, "body_count max_contacts 1"],
        lambda_t: Float[torch.Tensor, "body_count max_contacts 2 1"],
        gamma: Float[torch.Tensor, "body_count max_contacts 1"],
        J_n: Float[torch.Tensor, "body_count max_contacts 6"],
        J_t: Float[torch.Tensor, "body_count max_contacts 2 6"],
        v_n_prev: Float[torch.Tensor, "body_count max_contacts 1"],
        dt: float,
    ) -> None:
        """Verify the analytical Jacobian with finite differences.

        This function computes numerical Jacobian using finite differences
        and compares with the analytical Jacobian to check for errors.

        Args:
            model: Physical model
            state_in: Input state
            body_vel: Current body velocities
            lambda_n: Normal contact forces
            lambda_t: Tangential contact forces
            gamma: Auxiliary variables
            J_n: Normal contact Jacobians
            J_t: Tangential contact Jacobians
            v_n_prev: Previous normal velocities
            dt: Time step
            J_F: Analytical Jacobian to verify
        """
        C = state_in.contact_points.shape[1]  # Max contacts per body
        h = 1e-4  # Perturbation for finite differences
        total_dim = 6 + 4 * C  # Dimension of the Jacobian

        lambda_t1 = lambda_t[:, :, 0, :]  # [B, C, 1]
        lambda_t2 = lambda_t[:, :, 1, :]  # [B, C, 1]

        # Concatenate the variables
        x = torch.cat((body_vel, lambda_n, lambda_t1, lambda_t2, gamma), dim=1)

        # Compute the residuals
        res_d, res_n, res_fr1, res_fr2, res_frc = self.compute_residuals(
            model,
            state_in,
            body_vel,
            lambda_n,
            lambda_t1,
            gamma,
            J_n,
            J_t,
            v_n_prev,
            dt,
        )

        for i in range(total_dim):
            perturbation = torch.zeros_like(x)
            perturbation[:, i : i + 1] = h
            x_pert = x + perturbation
            res_d_pert, res_n_pert, res_fr1_pert, res_fr2_pert, res_frc_pert = (
                self.compute_residuals(
                    model,
                    state_in,
                    x_pert[:, :6],
                    x_pert[:, 6 : 6 + C],
                    torch.cat(
                        (
                            x_pert[:, 6 + C : 6 + 2 * C].unsqueeze(2),
                            x_pert[:, 6 + 2 * C : 6 + 3 * C].unsqueeze(2),
                        ),
                        dim=2,
                    ),
                    x_pert[:, 6 + 3 * C :],
                    J_n,
                    J_t,
                    v_n_prev,
                    dt,
                )
            )
            d_grad = ((res_d_pert - res_d) / h).squeeze(-1)
            n_grad = ((res_n_pert - res_n) / h).squeeze(-1)
            fr1_grad = ((res_fr1_pert - res_fr1) / h).squeeze(-1)
            fr2_grad = ((res_fr2_pert - res_fr2) / h).squeeze(-1)
            frc_grad = ((res_frc_pert - res_frc) / h).squeeze(-1)

            d_diff = d_grad - J_F[:, :6, i]
            n_diff = n_grad - J_F[:, 6 : 6 + C, i]
            fr1_diff = fr1_grad - J_F[:, 6 + C : 6 + 2 * C, i]
            fr2_diff = fr2_grad - J_F[:, 6 + 2 * C : 6 + 3 * C, i]
            frc_diff = frc_grad - J_F[:, 6 + 3 * C :, i]

            if torch.norm(d_diff) > 1e-2:
                print(f"Jacobian check failed for d: {torch.norm(d_diff)}")
            if torch.norm(n_diff) > 1e-2:
                print(f"Jacobian check failed for n: {torch.norm(n_diff)}")
            if torch.norm(fr1_diff) > 1e-2:
                print(f"Jacobian check failed for fr1: {torch.norm(fr1_diff)}")
            if torch.norm(fr2_diff) > 1e-2:
                print(f"Jacobian check failed for fr2: {torch.norm(fr2_diff)}")
            if torch.norm(frc_diff) > 1e-2:
                print(f"Jacobian check failed for frc: {torch.norm(frc_diff)}")

    def perform_newton_step(
        self,
        J_F: Float[torch.Tensor, "body_count total_dim total_dim"],
        F_x: Float[torch.Tensor, "body_count total_dim 1"],
        body_qd: Float[torch.Tensor, "body_count 6 1"],
        lambda_n: Float[torch.Tensor, "body_count max_contacts 1"],
        lambda_t: Float[torch.Tensor, "body_count max_contacts 2 1"],
        gamma: Float[torch.Tensor, "body_count max_contacts 1"],
    ) -> Tuple[
        Float[torch.Tensor, "body_count 6 1"],
        Float[torch.Tensor, "body_count max_contacts 1"],
        Float[torch.Tensor, "body_count max_contacts 2 1"],
        Float[torch.Tensor, "body_count max_contacts 1"]
    ]:
        """Perform one step of the Newton method.

        This function solves the linear system J_F @ Δx = -F_x and updates
        the variables: body_qd, lambda_n, lambda_t, and gamma.

        Args:
            J_F: Jacobian matrix
            F_x: Residual vector
            body_qd: Current body velocities
            lambda_n: Normal contact forces
            lambda_t: Tangential contact forces
            gamma: Auxiliary variables

        Returns:
            Updated variables (new_body_qd, new_lambda_n, new_lambda_t, new_gamma)
        """
        total_dim = J_F.shape[1]  # Dimension of the Jacobian
        C = lambda_n.shape[1]  # Maximum number of contacts

        # Regularize the Jacobian
        J_F_reg = J_F + self.reg * torch.eye(total_dim, device=self.device).unsqueeze(0)

        # Solve for the system of linear equations
        delta_x = torch.linalg.solve(J_F_reg, -F_x)  # [N, total_dim]

        new_body_qd = body_qd + delta_x[:, :6]
        new_lambda_n = lambda_n + delta_x[:, 6 : 6 + C]
        new_lambda_t1 = lambda_t[:, :, 0, :] + delta_x[:, 6 + C : 6 + 2 * C]
        new_lambda_t2 = lambda_t[:, :, 1, :] + delta_x[:, 6 + 2 * C : 6 + 3 * C]
        new_gamma = gamma + delta_x[:, 6 + 3 * C :]
        new_lambda_t = torch.cat((new_lambda_t1, new_lambda_t2), dim=2).unsqueeze(
            3
        )  # [N, C, 2, 1]

        return new_body_qd, new_lambda_n, new_lambda_t, new_gamma

    def perform_newton_step_line_search(
        self,
        model: Model,
        state_in: State,
        J_F: Float[torch.Tensor, "body_count total_dim total_dim"],
        F_x: Float[torch.Tensor, "body_count total_dim 1"],
        body_vel: Float[torch.Tensor, "body_count 6 1"],
        lambda_n: Float[torch.Tensor, "body_count max_contacts 1"],
        lambda_t: Float[torch.Tensor, "body_count max_contacts 2 1"],
        gamma: Float[torch.Tensor, "body_count max_contacts 1"],
        dt: float,
        J_n: Float[torch.Tensor, "body_count max_contacts 6"],
        J_t: Float[torch.Tensor, "body_count max_contacts 2 6"],
        v_n_prev: Float[torch.Tensor, "body_count max_contacts 1"],
    ) -> Tuple[
        Float[torch.Tensor, "body_count 6 1"],
        Float[torch.Tensor, "body_count max_contacts 1"],
        Float[torch.Tensor, "body_count max_contacts 2 1"],
        Float[torch.Tensor, "body_count max_contacts 1"]
    ]:
        """Perform a Newton step with line search to improve convergence.

        This function solves the linear system J_F·Δx = -F_x and applies a line
        search to find an appropriate step size that reduces the residual.

        Args:
            model: Physical model
            state_in: Input state
            J_F: Jacobian matrix
            F_x: Residual vector
            body_vel: Current body velocities
            lambda_n: Normal contact forces
            lambda_t: Tangential contact forces
            gamma: Auxiliary variables
            dt: Time step
            J_n: Normal contact Jacobians
            J_t: Tangential contact Jacobians
            v_n_prev: Previous normal velocities

        Returns:
            Updated variables (new_body_qd, new_lambda_n, new_lambda_t, new_gamma)
        """
        total_dim = J_F.shape[1]  # Dimension of the Jacobian
        C = lambda_n.shape[1]  # Maximum number of contacts

        # Regularize the Jacobian
        J_F_reg = J_F + self.reg * torch.eye(total_dim, device=self.device).unsqueeze(0)

        # Solve for the system of linear equations
        delta_x = torch.linalg.solve(J_F_reg, -F_x)

        # Extract deltas from delta_x
        delta_body_qd = delta_x[:, :6]  # [N, 6]
        delta_lambda_n = delta_x[:, 6 : 6 + C]  # [N, C]
        delta_lambda_t1 = delta_x[:, 6 + C : 6 + 2 * C]  # [N, C]
        delta_lambda_t2 = delta_x[:, 6 + 2 * C : 6 + 3 * C]  # [N, C]
        delta_gamma = delta_x[:, 6 + 3 * C :]  # [N, C]

        # Combine tangential deltas into a single tensor
        delta_lambda_t = torch.cat(
            (delta_lambda_t1.unsqueeze(2), delta_lambda_t2.unsqueeze(2)), dim=2
        )  # [N, C, 2, 1]

        # Compute current residual norm
        current_norm = torch.sum(F_x**2)

        # Line search parameters
        alpha = 1.0
        min_alpha = 1e-4
        max_linesearch_iters = 10

        for _ in range(max_linesearch_iters):
            # Compute trial updates
            body_qd_trial = body_vel + alpha * delta_body_qd
            lambda_n_trial = lambda_n + alpha * delta_lambda_n
            lambda_t_trial = lambda_t + alpha * delta_lambda_t
            gamma_trial = gamma + alpha * delta_gamma

            # Compute new residuals
            res_d_new, res_n_new, res_fr1_new, res_fr2_new, res_frc_new = (
                self.compute_residuals(
                    model,
                    state_in,
                    body_qd_trial,
                    lambda_n_trial,
                    lambda_t_trial,
                    gamma_trial,
                    J_n,
                    J_t,
                    v_n_prev,
                    dt,
                )
            )
            F_x_new = torch.cat(
                (res_d_new, res_n_new, res_fr1_new, res_fr2_new, res_frc_new), dim=1
            )
            new_norm = torch.sum(F_x_new**2)

            if new_norm < current_norm:
                break
            alpha /= 2.0
            if alpha < min_alpha:
                print("Linesearch failed to reduce residual; using smallest step")
                break

        # Apply final updates with chosen alpha
        new_body_qd = body_vel + alpha * delta_body_qd
        new_lambda_n = lambda_n + alpha * delta_lambda_n
        new_lambda_t = lambda_t + alpha * delta_lambda_t
        new_gamma = gamma + alpha * delta_gamma

        return new_body_qd, new_lambda_n, new_lambda_t, new_gamma

    def simulate(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        control: Control,
        dt: float,
    ):
        """Simulate one time step of the physics system with contact dynamics.

        This function updates the simulation state by solving the contact dynamics
        problem using a non-smooth Newton method.

        Args:
            model: Physical model with mass, friction parameters, etc.
            state_in: Input state with positions, velocities, and contacts
            state_out: Output state to be updated
            control: Control inputs for the simulation
            dt: Time step size
        """
        N = model.body_count  # Number of bodies
        C = state_in.contact_points.shape[1]  # Max contacts per body

        # Initial integration without contacts
        new_body_q, new_body_qd = self.integrator.integrate(
            body_q=state_in.body_q,
            body_qd=state_in.body_qd,
            body_f=state_in.body_f,
            body_inv_mass=model.body_inv_mass,
            body_inv_inertia=model.body_inv_inertia,
            gravity=model.gravity,
            dt=dt,
        )

        # Initialize contact forces
        # Compute contact Jacobians
        J_n = compute_contact_jacobians(
                    state_in.body_q,
                    state_in.contact_points,
                    state_in.contact_normals,
                    state_in.contact_mask
                )  # [body_count, max_contacts, 6]

                # Compute tangential contact Jacobians
        J_t = compute_tangential_jacobians(
                    state_in.body_q,
                    state_in.contact_points,
                    state_in.contact_normals
                )  # [body_count, max_contacts, 2, 6]
        v_n_prev = torch.bmm(J_n, state_in.body_qd)

        # Initialize variables
        lambda_n = torch.full((N, C, 1), 0.01, device=self.device)
        lambda_t = torch.full((N, C, 2, 1), 0.01, device=self.device)  # [N, C, 2, 1]
        gamma = torch.full((N, C, 1), 0.01, device=self.device)

        # Compute residuals
        res_d, res_n, res_fr1, res_fr2, res_frc = self.compute_residuals(
            model,
            state_in,
            new_body_qd,
            lambda_n,
            lambda_t,
            gamma,
            J_n,
            J_t,
            v_n_prev,
            dt,
        )
        F_x = torch.cat((res_d, res_n, res_fr1, res_fr2, res_frc), dim=1)

        # Newton iteration loop
        for i in range(self.iterations):
            res_d, res_n, res_fr1, res_fr2, res_frc = self.compute_residuals(
                model,
                state_in,
                new_body_qd,
                lambda_n,
                lambda_t,
                gamma,
                J_n,
                J_t,
                v_n_prev,
                dt,
            )

            F_x = torch.cat((res_d, res_n, res_fr1, res_fr2, res_frc), dim=1)
            J_F = self.compute_jacobian(
                model,
                state_in,
                new_body_qd,
                lambda_n,
                lambda_t,
                gamma,
                J_n,
                J_t,
                v_n_prev,
                dt,
            )

            new_body_qd, lambda_n, lambda_t, gamma = (
                self.perform_newton_step_line_search(
                    model,
                    state_in,
                    J_F,
                    F_x,
                    new_body_qd,
                    lambda_n,
                    lambda_t,
                    gamma,
                    dt,
                    J_n,
                    J_t,
                    v_n_prev,
                )
            )

        state_out.body_qd = new_body_qd
        state_out.time = state_in.time + dt

        # Semi-implicit Euler integration
        state_out.body_q[:, :3] = state_in.body_q[:, :3] + new_body_qd[:, 3:] * dt
        state_out.body_q[:, 3:] = integrate_quat_exact_batch(
            state_in.body_q[:, 3:], new_body_qd[:, :3], dt
        )
        # Print the residuals
        # print(f"Time: {state_out.time}, Residuals: {torch.norm(F_x)}")

        # if state_out.time >= 0.06:
        #     print("Stop")
