from os.path import curdir
from typing import Tuple

import torch

from pbd_torch.integrator import integrate_quat_exact_batch
from pbd_torch.integrator import SemiImplicitEulerIntegrator
from pbd_torch.model import Control
from pbd_torch.model import Model
from pbd_torch.model import State
from jaxtyping import Float
from pbd_torch.constraints import ContactConstraint
from pbd_torch.constraints import FrictionConstraint
from pbd_torch.constraints import DynamicsConstraint


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
        self.reg = 1e-12  # Regularization for Jacobian
        self.tol = 1e-8  # Tolerance for convergence

        self.contact_constraint = ContactConstraint(device=device)
        self.friction_constraint = FrictionConstraint(device=device)
        self.dynamics_constraint = DynamicsConstraint(
            mass_matrix=model.mass_matrix,
            g_accel=model.gravity,
            device=device
        )

        B = model.body_count
        C = model.max_contacts_per_body

        self._J_n = torch.zeros((B, C, 6), device=device)  # [B, C, 6]
        self._J_t = torch.zeros((B, 2 * C, 6), device=device)  # [B, 2 * C, 6]
        self._penetration_depth = torch.zeros((B, C, 1), device=device)  # [B, C, 1]
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
        dt: float,
    ) -> Tuple[
        Float[torch.Tensor, "body_count 6 1"],
        Float[torch.Tensor, "body_count max_contacts 1"],
        Float[torch.Tensor, "body_count 3 * max_contacts 1"],
    ]:
        # ------------------ Dynamics residual ------------------
        res_dynamics = self.dynamics_constraint.get_residuals(
            body_vel,
            self._body_vel_prev,
            lambda_n,
            lambda_t,
            self._body_force,
            self._J_n,
            self._J_t,
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

        return res_dynamics, res_contact, res_friction

    def compute_jacobian(
        self,
        body_vel: Float[torch.Tensor, "body_count 6 1"],
        lambda_n: Float[torch.Tensor, "body_count max_contacts 1"],
        lambda_t: Float[torch.Tensor, "body_count 2 * max_contacts 1"],
        gamma: Float[torch.Tensor, "body_count max_contacts 1"],
        dt: float,
    ) -> Float[torch.Tensor, "body_count total_dim total_dim"]:

        B = body_vel.shape[0]  # Body count
        C = lambda_n.shape[1]  # Maximum number of contacts per body
        total_dim = 6 + 4 * C
        J_F = torch.zeros((B, total_dim, total_dim), device=self.device)


        # Compute the derivatives of the dynamics constraint
        dres_d_dbody_vel, dres_d_dlambda_n, dres_d_dlambda_t = self.dynamics_constraint.get_derivatives(
            self._J_n,
            self._J_t,
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

        # Jacobian blocks
        # ------------------ Dynamics residual derivatives ------------------
        J_F[:, :6, :6] = dres_d_dbody_vel  # ∂res_d/∂body_vel
        J_F[:, :6, 6 : 6 + C] = dres_d_dlambda_n  # ∂res_d/∂lambda_n
        J_F[:, :6, 6 + C : 6 + 3 * C] = dres_d_dlambda_t  # ∂res_d/∂lambda_t

        # ------------------ Normal contact residual derivatives ------------------
        J_F[:, 6 : 6 + C, :6] = dres_n_dbody_vel  # ∂res_n/∂body_vel
        J_F[:, 6 : 6 + C, 6 : 6 + C] = dres_n_dlambda_n  # ∂res_n/∂lambda_n

        # ------------------ Friction contact residuals derivatives ------------------
        J_F[:, 6 + C : 6 + 4 * C, :6] = dres_t_dbody_vel # ∂res_t/∂body_vel
        J_F[:, 6 + C : 6 + 4 * C, 6 : 6 + C] = dres_t_dlambda_n # ∂res_t/∂lambda_n
        J_F[:, 6 + C : 6 + 4 * C, 6 + C : 6 + 3 * C] = dres_t_dlambda_t # ∂res_t/∂lambda_t
        J_F[:, 6 + C : 6 + 4 * C, 6 + 3 * C : 6 + 4 * C] = dres_t_dgamma # ∂res_t/∂gamma

        return J_F

    def compute_jacobian(
            self,
            body_vel: Float[torch.Tensor, "body_count 6 1"],
            lambda_n: Float[torch.Tensor, "body_count max_contacts 1"],
            lambda_t: Float[torch.Tensor, "body_count 2 * max_contacts 1"],
            gamma: Float[torch.Tensor, "body_count max_contacts 1"],
            dt: float,
    ) -> torch.Tensor:  # Returns a single sparse COO tensor
        B = body_vel.shape[0]
        C = lambda_n.shape[1]
        total_dim = 6 + 4 * C
        full_size = B * total_dim

        # Compute derivatives (already batched)
        dres_d_dbody_vel, dres_d_dlambda_n, dres_d_dlambda_t = self.dynamics_constraint.get_derivatives(
            self._J_n, self._J_t
        )  # Shapes: [B, 6, 6], [B, 6, C], [B, 6, 2*C]
        dres_n_dbody_vel, dres_n_dlambda_n = self.contact_constraint.get_derivatives(
            body_vel, self._body_vel_prev, lambda_n, self._J_n, self._penetration_depth,
            self._contact_mask, self.model.restitution, dt
        )  # Shapes: [B, C, 6], [B, C, C]
        dres_t_dbody_vel, dres_t_dlambda_n, dres_t_dlambda_t, dres_t_dgamma = self.friction_constraint.get_derivatives(
            body_vel, lambda_n, lambda_t, gamma, self._J_t, self._contact_mask, self.model.dynamic_friction
        )  # Shapes: [B, 3*C, 6], [B, 3*C, C], [B, 3*C, 2*C], [B, 3*C, C]

        # Lists to collect indices and values
        all_row_indices = []
        all_col_indices = []
        all_values = []

        # Base indices for each block
        body_offsets = torch.arange(B, device=self.device) * total_dim  # [B]

        # Helper function to add block to COO
        def add_block(values_tensor, row_start, col_start, row_size, col_size):
            # Find non-zero elements
            nonzero_mask = values_tensor != 0  # Shape [B, row_size, col_size]
            nonzero_values = values_tensor[nonzero_mask]  # Flattened non-zeros
            if nonzero_values.numel() == 0:
                return

            # Generate row and column indices
            row_indices_base, col_indices_base = torch.meshgrid(
                torch.arange(row_size, device=self.device),
                torch.arange(col_size, device=self.device),
                indexing='ij'
            )  # Shapes [row_size, col_size]
            row_indices_base = row_indices_base.expand(B, -1, -1)[nonzero_mask]  # Flattened
            col_indices_base = col_indices_base.expand(B, -1, -1)[nonzero_mask]  # Flattened

            # Add body offsets
            body_idx = torch.arange(B, device=self.device).view(-1, 1, 1).expand_as(values_tensor)[nonzero_mask]
            row_indices = body_offsets[body_idx] + row_start + row_indices_base
            col_indices = body_offsets[body_idx] + col_start + col_indices_base

            all_row_indices.append(row_indices)
            all_col_indices.append(col_indices)
            all_values.append(nonzero_values)

        # Populate blocks
        # Dynamics block (:6, :6)
        add_block(dres_d_dbody_vel, 0, 0, 6, 6)

        # Dynamics vs lambda_n (:6, 6:6+C)
        add_block(dres_d_dlambda_n, 0, 6, 6, C)

        # Dynamics vs lambda_t (:6, 6+C:6+3*C)
        add_block(dres_d_dlambda_t, 0, 6 + C, 6, 2 * C)

        # Contact block (6:6+C, :6)
        add_block(dres_n_dbody_vel, 6, 0, C, 6)

        # Contact vs lambda_n (6:6+C, 6:6+C)
        add_block(dres_n_dlambda_n, 6, 6, C, C)

        # Friction block (6+C:6+4*C, :6)
        add_block(dres_t_dbody_vel, 6 + C, 0, 3 * C, 6)

        # Friction vs lambda_n (6+C:6+4*C, 6:6+C)
        add_block(dres_t_dlambda_n, 6 + C, 6, 3 * C, C)

        # Friction vs lambda_t (6+C:6+4*C, 6+C:6+3*C)
        add_block(dres_t_dlambda_t, 6 + C, 6 + C, 3 * C, 2 * C)

        # Friction vs gamma (6+C:6+4*C, 6+3*C:6+4*C)
        add_block(dres_t_dgamma, 6 + C, 6 + 3 * C, 3 * C, C)

        # Combine all indices and values
        if not all_row_indices:  # Handle case with no non-zeros
            indices = torch.empty(2, 0, device=self.device, dtype=torch.long)
            values = torch.empty(0, device=self.device)
        else:
            indices = torch.stack([torch.cat(all_row_indices), torch.cat(all_col_indices)])  # Shape (2, nnz)
            values = torch.cat(all_values)  # Shape (nnz,)

        # Create sparse COO tensor
        J_F = torch.sparse_coo_tensor(indices, values, size=(full_size, full_size))
        return J_F

    def check_jacobian(
        self,
        J_F: Float[torch.Tensor, "body_count total_dim total_dim"],
        body_vel: Float[torch.Tensor, "body_count 6 1"],
        lambda_n: Float[torch.Tensor, "body_count max_contacts 1"],
        lambda_t: Float[torch.Tensor, "body_count 2 * max_contacts 1"],
        gamma: Float[torch.Tensor, "body_count max_contacts 1"],
        dt: float,
    ) -> None:
        C = lambda_n.shape[1]  # Max contacts per body
        h = 1e-6  # Perturbation for finite differences
        total_dim = 6 + 4 * C  # Dimension of the Jacobian

        # Concatenate the variables
        x = torch.cat((body_vel, lambda_n, lambda_t, gamma), dim=1)

        # Compute the residuals
        res_d, res_n, res_t = self.compute_residuals(
            body_vel,
            lambda_n,
            lambda_t,
            gamma,
            dt,
        )

        for i in range(total_dim):
            perturbation = torch.zeros_like(x)
            perturbation[:, i : i + 1] = h
            x_pert = x + perturbation
            res_d_pert, res_n_pert, res_t_pert = (
                self.compute_residuals(
                    x_pert[:, :6], # body_vel
                    x_pert[:, 6 : 6 + C], # lambda_n
                    x_pert[:, 6 + C : 6 + 3 * C], # lambda_t
                    x_pert[:, 6 + 3 * C :], # gamma
                    dt,
                )
            )
            d_grad = ((res_d_pert - res_d) / h).squeeze(-1)
            n_grad = ((res_n_pert - res_n) / h).squeeze(-1)
            t_grad = ((res_t_pert - res_t) / h).squeeze(-1)


            d_diff = d_grad - J_F[:, :6, i]
            n_diff = n_grad - J_F[:, 6 : 6 + C, i]
            t_diff = t_grad - J_F[:, 6 + C : 6 + 4 * C, i]

            if torch.norm(d_diff) > 1e-2:
                print(f"Jacobian check failed for d: {torch.norm(d_diff)} for var {i} ({self._debug_iter})")
            if torch.norm(n_diff) > 1e-2:
                print(f"Jacobian check failed for n: {torch.norm(n_diff)} for var {i} ({self._debug_iter})")
            if torch.norm(t_diff) > 1e-2:
                print(f"Jacobian check failed for t: {torch.norm(t_diff)} for var {i} ({self._debug_iter})")


    def perform_newton_step_line_search(
        self,
        J_F: Float[torch.Tensor, "body_count total_dim total_dim"],
        F_x: Float[torch.Tensor, "body_count total_dim 1"],
        body_vel: Float[torch.Tensor, "body_count 6 1"],
        lambda_n: Float[torch.Tensor, "body_count max_contacts 1"],
        lambda_t: Float[torch.Tensor, "body_count 2 * max_contacts 1"],
        gamma: Float[torch.Tensor, "body_count max_contacts 1"],
        dt: float,
    ) -> Tuple[
        Float[torch.Tensor, "body_count total_dim 1"],
        Float[torch.Tensor, "body_count 6 1"],
        Float[torch.Tensor, "body_count max_contacts 1"],
        Float[torch.Tensor, "body_count max_contacts 2 1"],
        Float[torch.Tensor, "body_count max_contacts 1"]
    ]:
        total_dim = J_F.shape[1]  # Dimension of the Jacobian
        C = lambda_n.shape[1]  # Maximum number of contacts

        # Regularize the Jacobian
        J_F_reg = J_F.to_dense() + self.reg * torch.eye(total_dim, device=self.device).unsqueeze(0)

        # Solve for the system of linear equations
        delta_x = torch.linalg.solve(J_F_reg, -F_x)

        # Extract deltas from delta_x
        delta_body_qd = delta_x[:, :6]  # [B, 6]
        delta_lambda_n = delta_x[:, 6 : 6 + C]  # [B, C]
        delta_lambda_t = delta_x[:, 6 + C : 6 + 3 * C]  # [B, 2 * C]
        delta_gamma = delta_x[:, 6 + 3 * C :]  # [B, C]

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

            # Compute new residuals
            res_d_new, res_n_new, res_t_new = (
                self.compute_residuals(
                    body_qd_trial,
                    lambda_n_trial,
                    lambda_t_trial,
                    gamma_trial,
                    dt,
                )
            )
            F_x_new = torch.cat([res_d_new, res_n_new, res_t_new], dim=1)
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

        return F_x, new_body_qd, new_lambda_n, new_lambda_t, new_gamma

    def simulate(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        dt: float,
    ):
        B = self.model.body_count  # Number of bodies
        C = state_in.contact_points.shape[1]  # Max contacts per body

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

        # Compute the penetration depth
        self._penetration_depth = self.contact_constraint.get_penetration_depths(
            state_in.body_q,
            state_in.contact_points,
            state_in.contact_points_ground,
            state_in.contact_normals,
        )  # [B, C, 1]

        # Initialize variables
        lambda_n = torch.full((B, C, 1), 0.01, device=self.device) # [B, C, 1]
        lambda_t = torch.full((B, 2 * C, 1), 0.01, device=self.device)  # [B, 2*C, 1]
        gamma = torch.full((B, C, 1), 0.01, device=self.device) # [B, C, 1]
        F_x_final = torch.zeros((B, 6 + 4 * C, 1), device=self.device) # [B, 6 + 4 * C, 1]

        # Newton iteration loop
        for i in range(self.iterations):
            res_d, res_n, res_t = self.compute_residuals(
                body_vel,
                lambda_n,
                lambda_t,
                gamma,
                dt,
            )

            F_x = torch.cat((res_d, res_n, res_t), dim=1)
            J_F = self.compute_jacobian(
                body_vel,
                lambda_n,
                lambda_t,
                gamma,
                dt,
            )


            F_x_final, body_vel, lambda_n, lambda_t, gamma = (
                self.perform_newton_step_line_search(
                    J_F,
                    F_x,
                    body_vel,
                    lambda_n,
                    lambda_t,
                    gamma,
                    dt,
                )
            )

            if torch.sum(F_x_final ** 2) < self.tol:
                break

            self._debug_iter += 1

        if torch.sum(F_x_final ** 2) > 0.1:
            print(f"Norm : {torch.sum(F_x_final ** 2)}, Time: {state_in.time + dt} ({self._debug_iter})")
            self._debug_F_x.append(F_x_final.unsqueeze(0))

        state_out.body_qd = body_vel
        state_out.time = state_in.time + dt

        # Semi-implicit Euler integration
        state_out.body_q[:, :3] = state_in.body_q[:, :3] + body_vel[:, 3:] * dt
        state_out.body_q[:, 3:] = integrate_quat_exact_batch(state_in.body_q[:, 3:], body_vel[:, :3], dt)
