from os.path import curdir
from pickle import DUP
from typing import Tuple

import torch

from pbd_torch.integrator import integrate_quat_exact_batch
from pbd_torch.integrator import SemiImplicitEulerIntegrator
from pbd_torch.model import Control
from pbd_torch.model import Model
from pbd_torch.model import State
from pbd_torch.ncp import ScaledFisherBurmeister
from pbd_torch.transform import transform_points_batch
from pbd_torch.transform import rotate_vectors_batch
from jaxtyping import Float
from pbd_torch.constraints import ContactConstraint
from pbd_torch.constraints import FrictionConstraint
from pbd_torch.constraints import DynamicsConstraint
from pbd_torch.constraints import RevoluteConstraint


class NonSmoothNewtonEngine:
    def __init__(self,
        model: Model,
        iterations: int = 10,
        device: torch.device = torch.device("cpu")
    ):
        self.model = model
        self.device = device
        self.iterations = iterations
        self.integrator = SemiImplicitEulerIntegrator(
            use_local_omega=False, device=device
        )
        self.reg = 1e-12 # Regularization for Jacobian
        self.tol = 1e-8  # Tolerance for convergence

        self.contact_constraint = ContactConstraint(device=device)
        self.friction_constraint = FrictionConstraint(device=device)
        self.dynamics_constraint = DynamicsConstraint(
            mass_matrix=model.mass_matrix,
            g_accel=model.gravity,
            device=device
        )
        self.joint_constraint = RevoluteConstraint(model=model, device=device)

        B = model.body_count
        C = model.max_contacts_per_body
        D = model.joint_mask_parent.shape[1]

        self._J_n = torch.zeros((B, C, 6), device=device)  # [B, C, 6]
        self._J_t = torch.zeros((B, 2 * C, 6), device=device)  # [B, 2 * C, 6]
        self._J_j = torch.zeros((B, 10 * D, 6), device=device)  # [B, 10 * C, 6]
        self._penetration_depth = torch.zeros((B, C, 1), device=device)  # [B, C, 1]
        self._body_trans = torch.zeros((B, 7, 1), device=device)  # [B, 7, 1]
        self._contact_mask = torch.zeros((B, C), device=device, dtype=torch.bool)  # [B, C]
        self._body_vel_prev = torch.zeros((B, 6, 1), device=device)  # [B, 6, 1]
        self._body_force = torch.zeros((B, 6, 1), device=device)  # [B, 6, 1]
        self._dt = None

        self._debug_iter = 0
        self._debug_F_x = []

    def compute_residuals(
        self,
        body_vel: Float[torch.Tensor, "body_count 6 1"],
        lambda_n: Float[torch.Tensor, "body_count max_contacts 1"],
        lambda_t: Float[torch.Tensor, "body_count 2 * max_contacts 1"],
        gamma: Float[torch.Tensor, "body_count max_contacts 1"],
        lambda_j: Float[torch.Tensor, "body_count 10 * max_joints 1"],
        dt: float,
    ) -> Tuple[
        Float[torch.Tensor, "body_count 6 1"],
        Float[torch.Tensor, "body_count max_contacts 1"],
        Float[torch.Tensor, "body_count 3 * max_contacts 1"],
        Float[torch.Tensor, "body_count 10 * max_joints 1"],
    ]:
        # ------------------ Dynamics residual ------------------
        res_dynamics = self.dynamics_constraint.get_residuals(
            body_vel,
            self._body_vel_prev,
            lambda_n,
            lambda_t,
            lambda_j,
            self._body_force,
            self._J_n,
            self._J_t,
            self._J_j,
            dt,
        ) # [B, 6, 1]

        # ------------------ Normal contact residual ------------------
        res_contact = self.contact_constraint.get_residuals(
            body_vel,
            self._body_vel_prev,
            lambda_n,
            self._J_n,
            self._penetration_depth,
            self._contact_mask,
            self.model.restitution,
            dt
        ) # [B, C, 1]

        # ------------------ Friction contact residuals ------------------
        res_friction = self.friction_constraint.get_residuals(
            body_vel,
            lambda_n,
            lambda_t,
            gamma,
            self._J_t,
            self._contact_mask,
            self.model.dynamic_friction
        ) # [B, 3 * C, 1]

        # ------------------ Joint contact residuals ------------------
        res_joint = self.joint_constraint.get_residuals(
            body_vel,
            lambda_j,
            self._body_trans,
            self._J_j,
            dt
        ) # [B, 10 * C, 1]

        return res_dynamics, res_contact, res_friction, res_joint

    def compute_jacobian(
        self,
        body_vel: Float[torch.Tensor, "body_count 6 1"],
        lambda_n: Float[torch.Tensor, "body_count max_contacts 1"],
        lambda_t: Float[torch.Tensor, "body_count 2 * max_contacts 1"],
        gamma: Float[torch.Tensor, "body_count max_contacts 1"],
        lambda_j: Float[torch.Tensor, "body_count 10 * max_joints 1"],
        dt: float,
    ) -> Float[torch.Tensor, "body_count total_dim total_dim"]:

        B = body_vel.shape[0]  # Body count
        C = lambda_n.shape[1]  # Maximum number of contacts per body
        D = lambda_j.shape[1] // 10  # Maximum number of joints per body
        total_dim = 6 + 4 * C + 10 * D
        J_F = torch.zeros((B, total_dim, total_dim), device=self.device)


        # Compute the derivatives of the dynamics constraint
        dres_d_dbody_vel, dres_d_dlambda_n, dres_d_dlambda_t, dres_d_dlambda_j = self.dynamics_constraint.get_derivatives(
            self._J_n,
            self._J_t,
            self._J_j,
        )

        # Compute the derivatives of the contact constraint
        dres_n_dbody_vel, dres_n_dlambda_n = self.contact_constraint.get_derivatives(
            body_vel,
            self._body_vel_prev,
            lambda_n,
            self._J_n,
            self._penetration_depth,
            self._contact_mask,
            self.model.restitution,
            dt
        )

        # Compute the derivatives of the friction constraint
        dres_t_dbody_vel, dres_t_dlambda_n, dres_t_dlambda_t, dres_t_dgamma = self.friction_constraint.get_derivatives(
            body_vel,
            lambda_n,
            lambda_t,
            gamma,
            self._J_t,
            self._contact_mask,
            self.model.dynamic_friction,
        )

        # Compute the derivative of the joint constraint
        dres_j_dbody_vel, dres_j_dlambda_j = self.joint_constraint.get_derivatives(self._J_j)

        # Jacobian blocks
        # ------------------ Dynamics residual derivatives ------------------
        J_F[:, :6, :6] = dres_d_dbody_vel  # ∂res_d/∂body_vel
        J_F[:, :6, 6 : 6 + C] = dres_d_dlambda_n  # ∂res_d/∂lambda_n
        J_F[:, :6, 6 + C : 6 + 3 * C] = dres_d_dlambda_t  # ∂res_d/∂lambda_t
        J_F[:, :6, 6 + 4 * C : 6 + 4 * C + 10 * D] = dres_d_dlambda_j # ∂res_d/∂lambda_j

        # ------------------ Normal contact residual derivatives ------------------
        J_F[:, 6 : 6 + C, :6] = dres_n_dbody_vel  # ∂res_n/∂body_vel
        J_F[:, 6 : 6 + C, 6 : 6 + C] = dres_n_dlambda_n  # ∂res_n/∂lambda_n

        # ------------------ Friction contact residuals derivatives ------------------
        J_F[:, 6 + C : 6 + 4 * C, :6] = dres_t_dbody_vel # ∂res_t/∂body_vel
        J_F[:, 6 + C : 6 + 4 * C, 6 : 6 + C] = dres_t_dlambda_n # ∂res_t/∂lambda_n
        J_F[:, 6 + C : 6 + 4 * C, 6 + C : 6 + 3 * C] = dres_t_dlambda_t # ∂res_t/∂lambda_t
        J_F[:, 6 + C : 6 + 4 * C, 6 + 3 * C : 6 + 4 * C] = dres_t_dgamma # ∂res_t/∂gamma

        # ------------------ Joint constraint derivatives ------------------
        J_F[:, 6 + 4 * C : 6 + 4 * C + 10 * D, :6] = dres_j_dbody_vel # ∂res_j/∂body_vel
        J_F[:, 6 + 4 * C : 6 + 4 * C + 10 * D, 6 + 4 * C : 6 + 4 * C + 10 * D] = dres_j_dlambda_j # ∂res_j/∂lambda_j
        return J_F

    def perform_newton_step_line_search(
        self,
        J_F: Float[torch.Tensor, "body_count total_dim total_dim"],
        F_x: Float[torch.Tensor, "body_count total_dim 1"],
        body_vel: Float[torch.Tensor, "body_count 6 1"],
        lambda_n: Float[torch.Tensor, "body_count max_contacts 1"],
        lambda_t: Float[torch.Tensor, "body_count 2 * max_contacts 1"],
        gamma: Float[torch.Tensor, "body_count max_contacts 1"],
        lambda_j: Float[torch.Tensor, "body_count 10 * max_joints 1"],
        dt: float,
    ) -> Tuple[
        Float[torch.Tensor, "body_count total_dim 1"],
        Float[torch.Tensor, "body_count 6 1"],
        Float[torch.Tensor, "body_count max_contacts 1"],
        Float[torch.Tensor, "body_count 2 * max_contacts 1"],
        Float[torch.Tensor, "body_count max_contacts 1"],
        Float[torch.Tensor, "body_count 10 * max_joints 1"],
    ]:
        total_dim = J_F.shape[1]  # Dimension of the Jacobian
        C = lambda_n.shape[1]  # Maximum number of contacts
        D = lambda_j.shape[1] // 10  # Maximum number of joints

        # Regularize the Jacobian
        J_F_reg = J_F + self.reg * torch.eye(total_dim, device=self.device).unsqueeze(0)

        # Solve for the system of linear equations
        delta_x = torch.linalg.solve(J_F_reg, -F_x)

        # Extract deltas from delta_x
        delta_body_qd = delta_x[:, :6]  # [B, 6]
        delta_lambda_n = delta_x[:, 6 : 6 + C]  # [B, C]
        delta_lambda_t = delta_x[:, 6 + C : 6 + 3 * C]  # [B, 2 * C]
        delta_gamma = delta_x[:, 6 + 3 * C : 6 + 4 * C]  # [B, C]
        delta_lambda_j = delta_x[:, 6 + 4 * C : 6 + 4 * C + 10 * D]  # [B, 10 * D]

        # Compute current residual norm
        norm = torch.sum(F_x**2)

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
            delta_lambda_j_trial = delta_lambda_j + alpha * delta_lambda_j

            # Compute new residuals
            res_d_new, res_n_new, res_t_new, res_j_new = self.compute_residuals(
                    body_qd_trial,
                    lambda_n_trial,
                    lambda_t_trial,
                    gamma_trial,
                    delta_lambda_j_trial,
                    dt,
                )

            F_x_new = torch.cat([res_d_new, res_n_new, res_t_new, res_j_new], dim=1)
            new_norm = torch.sum(F_x_new**2)

            if new_norm < norm:
                norm = new_norm
                F_x = F_x_new
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
        new_lambda_j = lambda_j + alpha * delta_lambda_j

        return F_x, new_body_qd, new_lambda_n, new_lambda_t, new_gamma, new_lambda_j

    def simulate(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        dt: float,
    ):
        B = self.model.body_count  # Number of bodies
        C = state_in.contact_points.shape[1]  # Max contacts per body
        D = self.model.joint_mask_parent.shape[1]  # Max dimensions per contact

        # Initial integration without contacts
        _, body_vel = self.integrator.integrate(
            body_q=state_in.body_q,
            body_qd=state_in.body_qd,
            body_f=state_in.body_f,
            body_inv_mass=self.model.body_inv_mass,
            body_inv_inertia=self.model.body_inv_inertia,
            gravity=self.model.gravity,
            dt=dt,
        )

        self._body_trans = state_in.body_q.clone()
        self._body_force = state_in.body_f.clone()
        self._body_vel_prev = state_in.body_qd.clone()
        self._contact_mask = state_in.contact_mask.clone()

        # Compute normal contact Jacobians
        self._J_n = self.contact_constraint.compute_contact_jacobians(
            state_in.body_q,
            state_in.contact_points,
            state_in.contact_normals,
            state_in.contact_mask
        )  # [B, C, 6]

        # Compute tangential contact Jacobians
        self._J_t = self.friction_constraint.compute_tangential_jacobians(
            state_in.body_q,
            state_in.contact_points,
            state_in.contact_normals,
            state_in.contact_mask
        )  # [B, 2 * C, 6]

        # Compute joint Jacobians
        self._J_j = self.joint_constraint.compute_jacobians(
            state_in.body_q,
        )  # [B, 10 * D, 6]

        # Compute the penetration depth
        self._penetration_depth = self.contact_constraint.get_penetration_depths(
            state_in.body_q,
            state_in.contact_points,
            state_in.contact_points_ground,
            state_in.contact_normals,
        )  # [B, C, 1]

        # Initialize variables
        lambda_n = torch.full((B, C, 1), 0.01, device=self.device) # [B, C, 1]
        lambda_t = torch.full((B, 2 * C, 1), 0.01, device=self.device)  # [B, 2 * C, 1]
        gamma = torch.full((B, C, 1), 0.01, device=self.device) # [B, C, 1]
        lambda_j = torch.full((B, 10 * D, 1), 0.01, device=self.device) # [B, 10 * D, 1]
        F_x_final = torch.zeros((B, 6 + 4 * C, 1), device=self.device) # [B, 6 + 4 * C, 1]

        # Newton iteration loop
        for i in range(self.iterations):
            res_d, res_n, res_t, res_j = self.compute_residuals(
                body_vel,
                lambda_n,
                lambda_t,
                gamma,
                lambda_j,
                dt,
            )

            F_x = torch.cat((res_d, res_n, res_t, res_j), dim=1)
            J_F = self.compute_jacobian(
                body_vel,
                lambda_n,
                lambda_t,
                gamma,
                lambda_j,
                dt,
            )


            F_x_final, body_vel, lambda_n, lambda_t, gamma, lambda_j = self.perform_newton_step_line_search(
                    J_F,
                    F_x,
                    body_vel,
                    lambda_n,
                    lambda_t,
                    gamma,
                    lambda_j,
                    dt,
                )

            if torch.sum(F_x_final ** 2) < self.tol:
                break

            self._debug_iter += 1

        if torch.sum(F_x_final ** 2) > 0.1:
            print(f"Norm : {torch.sum(F_x_final ** 2)}, Time: {state_in.time + dt} ({self._debug_iter})")
            self._debug_F_x.append(F_x_final.unsqueeze(0))
            # print(f"F_x_final shape: {F_x_final.shape}")

        state_out.body_qd = body_vel
        state_out.time = state_in.time + dt

        # Semi-implicit Euler integration
        state_out.body_q[:, :3] = state_in.body_q[:, :3] + body_vel[:, 3:] * dt
        state_out.body_q[:, 3:] = integrate_quat_exact_batch(state_in.body_q[:, 3:], body_vel[:, :3], dt)
