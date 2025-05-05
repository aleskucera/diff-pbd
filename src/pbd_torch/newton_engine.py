import os
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from pbd_torch.constraints import ContactConstraint
from pbd_torch.constraints import DynamicsConstraint
from pbd_torch.constraints import FrictionConstraint
from pbd_torch.constraints import RevoluteConstraint
from pbd_torch.integrator import integrate_quat_exact_batch
from pbd_torch.integrator import SemiImplicitEulerIntegrator
from pbd_torch.model import Control
from pbd_torch.model import Model
from pbd_torch.model import State
from xitorch.optimize import rootfinder
from pbd_torch.utils import forces_from_joint_acts



class NonSmoothNewtonEngine:
    """
    A non-smooth Newton engine for solving dynamics and constraints in physics-based simulations.

    Attributes:
        model (Model): The physical model containing body and joint information.
        device (torch.device): The device (CPU/GPU) on which computations are performed.
        iterations (int): The maximum number of Newton iterations to perform.
        integrator (SemiImplicitEulerIntegrator): The integrator for updating body states.

        reg (float): Regularization term for the Jacobian.
        tol (float): Tolerance for convergence in the Newton solver.

        contact_constraint (ContactConstraint): Constraint for resolving contact collisions.
        friction_constraint (FrictionConstraint): Constraint for resolving friction forces.
        dynamics_constraint (DynamicsConstraint): Constraint for enforcing dynamics.
        revolute_constraint (RevoluteConstraint): Constraint for handling revolute joints.

        debug_jacobian (bool): Flag to enable Jacobian validation.
        debug_folder (str): Directory for saving debug visualizations.

        _J_n (torch.Tensor): Jacobian for normal contact forces [B, C, 6].
        _J_t (torch.Tensor): Jacobian for tangential contact forces [B, 2_C, 6].
        _J_j_p (torch.Tensor): Jacobian for parent joints [5_D, 6].
        _J_j_c (torch.Tensor): Jacobian for child joints [5_D, 6].
        _body_trans (torch.Tensor): Body transformations [B, 7, 1].
        _penetration_depth (torch.Tensor): Penetration depths for contacts [B, C, 1].
        _contact_mask (torch.Tensor): Mask indicating active contacts [B, C].
        _body_vel_prev (torch.Tensor): Previous body velocities [B, 6, 1].
        _body_force (torch.Tensor): External forces applied to bodies [B, 6, 1].
        _dt (float): Time step for simulation.
        _debug_iter (int): Debug iteration counter.
        _debug_F_x (list): List to store residuals for debugging.
    """

    def __init__(
        self,
        model: Model,
        iterations: int = 10,
        regularization: float = 1e-6,
        convergence_tolerance: float = 1e-5,
        debug_jacobian: bool = False,
        debug_folder: str = "debug",
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.device = device
        self.iterations = iterations
        self.integrator = SemiImplicitEulerIntegrator(
            use_local_omega=False, device=device
        )
        self.reg = regularization
        self.tol = convergence_tolerance

        self.debug_jacobian = debug_jacobian
        self.debug_folder = debug_folder

        self.contact_constraint = ContactConstraint(device=device)
        self.friction_constraint = FrictionConstraint(device=device)
        self.dynamics_constraint = DynamicsConstraint(model)
        self.revolute_constraint = RevoluteConstraint(model)

        B = model.body_count
        C = model.max_contacts_per_body
        D = model.joint_count

        self._J_n = torch.zeros((B, C, 6), device=device)  # [B, C, 6]
        self._J_t = torch.zeros((B, 2 * C, 6), device=device)  # [B, 2C, 6]
        self._J_j_p = torch.zeros((5 * D, 6), device=device)  # [5D, 6]
        self._J_j_c = torch.zeros((5 * D, 6), device=device)  # [5D, 6]
        self._body_trans = torch.zeros((B, 7, 1), device=device)  # [B, 7, 1]
        self._penetration_depth = torch.zeros((B, C, 1), device=device)  # [B, C, 1]
        self._contact_mask = torch.zeros((B, C), device=device, dtype=torch.bool)  # [B, C]
        self._contact_weight = torch.zeros((B, C), device=device, dtype=torch.float) # [B, C]
        self._contact_mask_n = torch.zeros((B, C), device=device, dtype=torch.bool)  # [B, C]
        self._contact_mask_t = torch.zeros((B, C), device=device, dtype=torch.bool)  # [B, C]
        self._body_vel_prev = torch.zeros((B, 6, 1), device=device)  # [B, 6, 1]
        self._body_force = torch.zeros((B, 6, 1), device=device)  # [B, 6, 1]
        self._dt = None

        self._debug_iter = 0
        self._debug_F_x = []

    def _compute_attributes(self, state_in: State, control: Control, dt: float) -> None:
        control_body_f = forces_from_joint_acts(control.joint_act,
                                                state_in.joint_q,
                                                state_in.joint_qd,
                                                self.model.joint_ke,
                                                self.model.joint_kd,
                                                state_in.body_q,
                                                self.model.joint_parent,
                                                self.model.joint_child,
                                                self.model.joint_X_p,
                                                self.model.joint_X_c)

        self._dt = dt
        self._body_force = state_in.body_f + control_body_f  # [B, 6, 1]
        self._body_trans = state_in.body_q.clone()  # [B, 7, 1]
        self._body_vel_prev = state_in.body_qd.clone()  # [B, 6, 1]
        self._contact_mask = state_in.contact_mask_per_body.clone()  # [B, C]
        self._contact_weight = state_in.contact_weight_per_body.clone()  # [B, C]

        # Compute normal contact Jacobians
        self._J_n = self.contact_constraint.compute_contact_jacobians(
            state_in.body_q,
            state_in.contact_points_per_body,
            state_in.contact_normals_per_body,
            state_in.contact_mask_per_body,
        )  # [B, C, 6]

        # Compute tangential contact Jacobians
        self._J_t = self.friction_constraint.compute_tangential_jacobians(
            state_in.body_q,
            state_in.contact_points_per_body,
            state_in.contact_normals_per_body,
            state_in.contact_mask_per_body,
        )  # [B, 2C, 6]

        # Compute the penetration depth
        self._penetration_depth = self.contact_constraint.get_penetration_depths(
            state_in.body_q,
            state_in.contact_points_per_body,
            state_in.contact_points_ground_per_body,
            state_in.contact_normals_per_body,
        )  # [B, C, 1]

        self._J_j_p, self._J_j_c = self.revolute_constraint.compute_jacobians(
            self._body_trans
        )  # [5D, 6], [5D, 6]

    def flatten_variables(
        self,
        body_vel: Float[torch.Tensor, "B 6 1"],
        lambda_n: Float[torch.Tensor, "B C 1"],
        lambda_t: Float[torch.Tensor, "B 2C 1"],
        gamma: Float[torch.Tensor, "B C 1"],
        lambda_j: Float[torch.Tensor, "5D 1"],
    ) -> Float[torch.Tensor, "B(6 + 4C) + 5D 1"]:
        x_batched = torch.cat(
            [body_vel, lambda_n, lambda_t, gamma], dim=1
        )  # [B, 6 + C + 2C + C, 1]
        x_batched_flat = x_batched.view(-1, 1)  # [B(6 + 4C), 1]
        x = torch.cat([x_batched_flat, lambda_j], dim=0)  # [B(6 + 4C) + 5D, 1]
        return x

    def unflatten_variables(
        self,
        x: Float[torch.Tensor, "B(6 + 4C) + 5D 1"],
    ) -> Tuple[
        Float[torch.Tensor, "B 6 1"],
        Float[torch.Tensor, "B C 1"],
        Float[torch.Tensor, "B 2C 1"],
        Float[torch.Tensor, "B C 1"],
        Float[torch.Tensor, "5D 1"],
    ]:
        B = self.model.body_count
        C = self.model.max_contacts_per_body
        D = self.model.joint_count
        total_dim_batched = 6 + 4 * C
        x_batched_flat = x[: B * total_dim_batched].view(
            B, total_dim_batched, 1
        )  # [B, 6 + 4C, 1]
        body_vel = x_batched_flat[:, :6, :]  # [B, 6, 1]
        lambda_n = x_batched_flat[:, 6 : 6 + C, :]  # [B, C, 1]
        lambda_t = x_batched_flat[:, 6 + C : 6 + 3 * C, :]  # [B, 2C, 1]
        gamma = x_batched_flat[:, 6 + 3 * C :, :]  # [B, C, 1]
        lambda_j = x[B * total_dim_batched :].view(5 * D, 1)  # [5D, 1]
        return body_vel, lambda_n, lambda_t, gamma, lambda_j

    def flatten_residuals(
        self,
        res_d: Float[torch.Tensor, "B 6 1"],
        res_n: Float[torch.Tensor, "B C 1"],
        res_t: Float[torch.Tensor, "B 3C 1"],
        res_j: Float[torch.Tensor, "5D 1"]
    ) -> Float[torch.Tensor, "B(6 + 4C) + 5D 1"]:
        res_batched = torch.cat((res_d, res_n, res_t), dim=1)  # [B, 6 + C + 3C, 1]
        res = torch.cat((res_batched.view(-1, 1), res_j), dim=0)  # [B(6 + 4C) + 5D, 1]
        return res

    def unflatten_residuals(self,
        res: Float[torch.Tensor, "B(6 + 4C) + 5D 1"],
    ) -> Tuple[
        Float[torch.Tensor, "B 6 1"],
        Float[torch.Tensor, "B C 1"],
        Float[torch.Tensor, "B 3C 1"],
        Float[torch.Tensor, "5D 1"],
    ]:
        B = self.model.body_count
        C = self.model.max_contacts_per_body
        D = self.model.joint_count
        total_dim_batched = 6 + 4 * C
        res_batched_flat = res[:B * total_dim_batched].view(
            B, total_dim_batched, 1
        )
        res_d = res_batched_flat[:, :6, :]  # [B, 6, 1]
        res_n = res_batched_flat[:, 6 : 6 + C, :]  # [B, C, 1]
        res_t = res_batched_flat[:, 6 + C : 6 + 3 * C, :]  # [B, 3C, 1]
        res_j = res[B * total_dim_batched :].view(5 * D, 1)  # [5D, 1]
        return res_d, res_n, res_t, res_j

    # Inside NonSmoothNewtonEngine class
    def residual(
        self,
        x: Float[torch.Tensor, "B(6 + 4C) + 5D 1"],
        body_trans: Float[torch.Tensor, "B 7 1"] = None,
        body_vel_prev: Float[torch.Tensor, "B 6 1"] = None,
        body_force: Float[torch.Tensor, "B 6 1"] = None,
        J_n: Float[torch.Tensor, "B C 6"] = None,
        J_t: Float[torch.Tensor, "B 2C 6"] = None,
        J_j_p: Float[torch.Tensor, "5D 6"] = None,
        J_j_c: Float[torch.Tensor, "5D 6"] = None,
        penetration_depth: Float[torch.Tensor, "B C 1"] = None,
        contact_weight: Float[torch.Tensor, "B C 1"] = None,
        restitution: Float[torch.Tensor, "B C 1"] = None,
        dynamic_friction: Float[torch.Tensor, "B C 1"] = None,
    ) -> Float[torch.Tensor, "B(6 + 4C) + 5D 1"]:
        # Use default arguments only if parameters are not explicitly provided
        body_trans = body_trans if body_trans is not None else self._body_trans
        body_vel_prev = body_vel_prev if body_vel_prev is not None else self._body_vel_prev
        body_force = body_force if body_force is not None else self._body_force
        J_n = J_n if J_n is not None else self._J_n
        J_t = J_t if J_t is not None else self._J_t
        J_j_p = J_j_p if J_j_p is not None else self._J_j_p
        J_j_c = J_j_c if J_j_c is not None else self._J_j_c
        penetration_depth = (
            penetration_depth if penetration_depth is not None else self._penetration_depth
        )
        contact_weight = contact_weight if contact_weight is not None else self._contact_weight
        restitution = restitution if restitution is not None else self.model.restitution
        dynamic_friction = (
            dynamic_friction if dynamic_friction is not None else self.model.dynamic_friction
        )

        body_vel, lambda_n, lambda_t, gamma, lambda_j = self.unflatten_variables(x)

        # ------------------ Dynamics residual ------------------
        # Use passed arguments instead of self._...
        res_dynamics = self.dynamics_constraint.get_residuals(
            body_vel,
            body_vel_prev,
            lambda_n,
            lambda_t,
            lambda_j,
            body_force,
            J_n,
            J_t,
            J_j_p,
            J_j_c,
            self._dt,
        )

        # ------------------ Normal contact residual ------------------
        res_contact = self.contact_constraint.get_residuals(
            body_vel,
            body_vel_prev,
            lambda_n,
            J_n,
            penetration_depth,
            self._contact_mask,
            contact_weight,
            restitution,
            self._dt,
        )

        # ------------------ Friction contact residuals ------------------
        res_friction = self.friction_constraint.get_residuals(
            body_vel,
            lambda_n,
            lambda_t,
            gamma,
            J_t,
            self._contact_mask,
            dynamic_friction,
        )

        # ---------------- Joint residuals ------------------
        res_joint = self.revolute_constraint.get_residuals(
            body_vel,
            body_trans,
            J_j_p,
            J_j_c,
            self._dt
        )

        res = self.flatten_residuals(res_dynamics, res_contact, res_friction, res_joint)

        return res

    def compute_jacobian(
        self,
        x: Float[torch.Tensor, "B(6 + 4C) + 5D 1"],
    ) -> Float[torch.Tensor, "B(6 + 4C) + 5D B(6 + 4C) + 5D"]:
        body_vel, lambda_n, lambda_t, gamma, lambda_j = self.unflatten_variables(x)

        B = self.model.body_count  # Body count
        C = self.model.max_contacts_per_body  # Max contacts per body
        D = self.model.joint_count  # Joint count
        total_dim_batched = 6 + 4 * C  # Per-body variables
        full_size = (
            B * total_dim_batched + 5 * D
        )  # Total size including unbatched lambda_j

        # Compute derivatives (already batched)
        dres_d_dbody_vel, dres_d_dlambda_n, dres_d_dlambda_t, dres_d_dlambda_j = (
            self.dynamics_constraint.get_derivatives(
                self._J_n, self._J_t, self._J_j_p, self._J_j_c
            )
        )  # Shapes: [B, 6, 6], [B, 6, C], [B, 6, 2C], [B, 6, 5D]

        dres_n_dbody_vel, dres_n_dlambda_n = self.contact_constraint.get_derivatives(
            body_vel,
            self._body_vel_prev,
            lambda_n,
            self._J_n,
            self._penetration_depth,
            self._contact_mask,
            self._contact_weight,
            self.model.restitution,
            self._dt,
        )  # Shapes: [B, C, 6], [B, C, C]

        dres_t_dbody_vel, dres_t_dlambda_n, dres_t_dlambda_t, dres_t_dgamma = (
            self.friction_constraint.get_derivatives(
                body_vel,
                lambda_n,
                lambda_t,
                gamma,
                self._J_t,
                self._contact_mask,
                self.model.dynamic_friction,
            )
        )  # Shapes: [B, 3C, 6], [B, 3C, C], [B, 3C, 2C], [B, 3C, C]

        dres_j_dbody_vel = self.revolute_constraint.get_derivatives(
            body_vel, self._J_j_p, self._J_j_c
        )  # Shapes: [5D, 6]

        # Lists to collect indices and values
        all_row_indices = []
        all_col_indices = []
        all_values = []

        # Base indices for each block
        body_offsets = torch.arange(B, device=self.device) * total_dim_batched  # [B]

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
                indexing="ij",
            )  # Shapes [row_size, col_size]
            row_indices_base = row_indices_base.expand(B, -1, -1)[
                nonzero_mask
            ]  # Flattened
            col_indices_base = col_indices_base.expand(B, -1, -1)[
                nonzero_mask
            ]  # Flattened

            # Add body offsets
            body_idx = (
                torch.arange(B, device=self.device)
                .view(-1, 1, 1)
                .expand_as(values_tensor)[nonzero_mask]
            )
            row_indices = body_offsets[body_idx] + row_start + row_indices_base
            col_indices = body_offsets[body_idx] + col_start + col_indices_base

            all_row_indices.append(row_indices)
            all_col_indices.append(col_indices)
            all_values.append(nonzero_values)

        # Dynamics wrt. body_vel (:6, :6)
        add_block(dres_d_dbody_vel, 0, 0, 6, 6)

        # Dynamics wrt. lambda_n (:6, 6:6+C)
        add_block(dres_d_dlambda_n, 0, 6, 6, C)

        # Dynamics wrt. lambda_t (:6, 6+C:6+3C)
        add_block(dres_d_dlambda_t, 0, 6 + C, 6, 2 * C)

        # Dynamics wrt. lambda_j (dynamics contribution)
        nonzero_mask = dres_d_dlambda_j != 0  # [B, 6, 5D]
        if nonzero_mask.any():
            nonzero_values = dres_d_dlambda_j[nonzero_mask]
            row_indices_base, col_indices_base = torch.where(
                nonzero_mask.view(B * 6, 5 * D)
            )
            body_idx = row_indices_base // 6
            row_indices = body_offsets[body_idx] + (row_indices_base % 6)
            col_indices = col_indices_base + B * total_dim_batched
            all_row_indices.append(row_indices)
            all_col_indices.append(col_indices)
            all_values.append(nonzero_values)

        # Contact wrt. body_vel (6:6+C, :6)
        add_block(dres_n_dbody_vel, 6, 0, C, 6)

        # Contact vs lambda_n (6:6+C, 6:6+C)
        add_block(dres_n_dlambda_n, 6, 6, C, C)

        # Friction wrt. body_vel (6+C:6+4C, :6)
        add_block(dres_t_dbody_vel, 6 + C, 0, 3 * C, 6)

        # Friction wrt. lambda_n (6+C:6+4C, 6:6+C)
        add_block(dres_t_dlambda_n, 6 + C, 6, 3 * C, C)

        # Friction wrt. lambda_t (6+C:6+4C, 6+C:6+3C)
        add_block(dres_t_dlambda_t, 6 + C, 6 + C, 3 * C, 2 * C)

        # Friction wrt. gamma (6+C:6+4C, 6+3C:6+4C)
        add_block(dres_t_dgamma, 6 + C, 6 + 3 * C, 3 * C, C)

        # Joint block: ∂res_j/∂v
        nonzero_mask = dres_j_dbody_vel != 0  # [5D, 6B]
        if nonzero_mask.any():
            nonzero_values = dres_j_dbody_vel[nonzero_mask]
            row_indices_base, col_indices_base = torch.where(nonzero_mask)
            row_indices = row_indices_base + B * total_dim_batched
            body_idx = col_indices_base // 6
            col_indices = body_offsets[body_idx] + (col_indices_base % 6)
            all_row_indices.append(row_indices)
            all_col_indices.append(col_indices)
            all_values.append(nonzero_values)

        # Combine all indices and values
        if not all_row_indices:  # Handle case with no non-zeros
            indices = torch.empty(2, 0, device=self.device, dtype=torch.long)
            values = torch.empty(0, device=self.device)
        else:
            indices = torch.stack(
                [torch.cat(all_row_indices), torch.cat(all_col_indices)]
            )  # Shape (2, nnz)
            values = torch.cat(all_values)  # Shape (nnz,)

        # Create sparse COO tensor
        J_F = torch.sparse_coo_tensor(indices, values, size=(full_size, full_size))

        return J_F

    def compute_numerical_jacobian(
        self,
        x: Float[torch.Tensor, "B(6 + 4C) + 5D 1"],
        eps: float = 1e-6,
    ) -> Float[torch.Tensor, "B(6 + 4C) + 5D B(6 + 4C) + 5D"]:
        B = self.model.body_count  # Body count
        C = self.model.max_contacts_per_body  # Max contacts per body
        D = self.model.joint_count  # Joint count
        total_dim_batched = 6 + 4 * C  # Body velocities + contact variables
        full_size = B * total_dim_batched + 5 * D  # Total variables including joints

        # Flatten the variables into a single vector
        F_x = self.residual(x)

        # Initialize numerical Jacobian
        J_num = torch.zeros((full_size, full_size), device=self.device)

        # Compute each column of the Jacobian
        for i in range(full_size):
            x_plus = x.clone()
            x_plus[i, 0] += eps  # Perturb the i-th variable
            F_x_plus = self.residual(x_plus)
            J_num[:, i] = (F_x_plus - F_x).squeeze() / eps

        return J_num

    def visualize_jacobian_comparison(
        self,
        J_ana: Float[torch.Tensor, "B(6 + 4C) + 5D B(6 + 4C) + 5D"],
        J_num: Float[torch.Tensor, "B(6 + 4C) + 5D B(6 + 4C) + 5D"],
        iteration: int,
    ) -> None:
        """
        Visualizes the comparison between the analytical and numerical Jacobians.

        Args:
            J_ana: Analytical Jacobian (sparse COO tensor).
            J_num: Numerical Jacobian [B(6 + 4C) + 5D, B(6 + 4C) + 5D].
            iteration: Current iteration number for debugging.
        """
        # Convert tensors to numpy arrays for plotting
        if J_ana.is_sparse:
            J_ana_dense = J_ana.to_dense().cpu().numpy()
        else:
            J_ana_dense = J_ana.cpu().numpy()
        J_num_np = J_num.cpu().numpy()

        # Compute absolute difference
        diff = np.abs(J_ana_dense - J_num_np)

        # Create a figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot Analytical Jacobian
        im0 = axes[0].imshow(
            J_ana_dense, aspect="auto", cmap="viridis", interpolation="nearest"
        )
        axes[0].set_title("Analytical Jacobian")
        plt.colorbar(im0, ax=axes[0])

        # Plot Numerical Jacobian
        im1 = axes[1].imshow(
            J_num_np, aspect="auto", cmap="viridis", interpolation="nearest"
        )
        axes[1].set_title("Numerical Jacobian")
        plt.colorbar(im1, ax=axes[1])

        # Plot Absolute Difference with Threshold
        threshold = 0.1  # Adjust this value as needed
        diff_thresholded = np.copy(diff)
        diff_thresholded[diff_thresholded < threshold] = (
            0  # Highlight only significant differences
        )
        im2 = axes[2].imshow(
            diff_thresholded, aspect="auto", cmap="hot", interpolation="nearest"
        )
        axes[2].set_title(f"Absolute Difference (Threshold: {threshold})")
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()

        # Save the figure to the run folder
        os.makedirs(self.debug_folder, exist_ok=True)
        fig_path = os.path.join(
            self.debug_folder, f"jacobian_comparison_iter_{iteration}.png"
        )
        plt.savefig(fig_path)
        plt.close()

        # Compute and display the maximum absolute difference
        max_diff = np.max(diff)
        print(
            f"Iteration {iteration} - Maximum absolute difference in Jacobian: {max_diff:.6f}"
        )

    def perform_newton_step_line_search(
        self,
        F_x: Float[torch.Tensor, "B(6 + 4C) + 5D 1"],
        J_F: Float[torch.Tensor, "B(6 + 4C) + 5D B(6 + 4C) + 5D"],
        x: Float[torch.Tensor, "B(6 + 4C) + 5D 1"],
    ) -> Tuple[
        Float[torch.Tensor, "B(6 + 4C) + 5D 1"],
        Float[torch.Tensor, "B(6 + 4C) + 5D 1"],
    ]:
        B = self.model.body_count  # Body count
        C = self.model.max_contacts_per_body  # Max contacts per body
        D = self.model.joint_count  # Joint count
        total_dim_batched = 6 + 4 * C  # Per-body variables
        full_size = (
            B * total_dim_batched + 5 * D
        )  # Total size including unbatched lambda_j

        # Regularize the sparse Jacobian
        reg_indices = torch.arange(full_size, device=self.device)  # [full_size]
        reg_indices = torch.stack([reg_indices, reg_indices])  # [2, full_size]
        reg_values = torch.full(
            (full_size,), self.reg, device=self.device
        )  # [full_size]
        reg_matrix = torch.sparse_coo_tensor(
            reg_indices, reg_values, size=(full_size, full_size)
        )
        J_F_reg = J_F + reg_matrix

        J_F_reg_dense = J_F_reg.to_dense()  # [full_size, full_size]
        delta_x = torch.linalg.solve(J_F_reg_dense, -F_x)  # [full_size, 1]

        # Clamp the detlta_x to avoid huge jumps
        delta_x = torch.clamp(delta_x, -100.0, 100.0)

        norm = torch.sum(F_x**2)  # Scalar
        alpha = 1.0
        min_alpha = 1e-4
        max_linesearch_iters = 10

        for _ in range(max_linesearch_iters):
            x_trial = x + alpha * delta_x  # [B(6 + 4C) + 5D, 1]

            F_x_new = self.residual(x_trial)  # [B(6 + 4C) + 5D, 1]
            norm_new = torch.sum(F_x_new**2)  # Scalar

            if norm_new < norm:
                F_x = F_x_new
                break
            alpha /= 2.0
            if alpha < min_alpha:
                print("Linesearch failed to reduce residual; using smallest step")
                break

        new_x = x + alpha * delta_x

        return F_x, new_x

    def newton(
        self,
        x0: Float[torch.Tensor, "B(6 + 4C) + 5D 1"],
    ) -> Float[torch.Tensor, "B(6 + 4C) + 5D 1"]:

        B = self.model.body_count  # Body count
        C = self.model.max_contacts_per_body
        D = self.model.joint_count
        total_dim_batched = 6 + 4 * C
        full_size = B * total_dim_batched + 5 * D

        x = x0.clone()  # [B(6 + 4C) + 5D, 1]
        F_x_final = torch.zeros((full_size, 1), device=self.device)

        # Newton iteration loop
        for i in range(self.iterations):
            F_x = self.residual(x)
            J_F = self.compute_jacobian(x)

            # Perform Jacobian check only in the first iteration
            if self.debug_jacobian and i == 0:
                J_num = self.compute_numerical_jacobian(x)
                self.visualize_jacobian_comparison(J_F, J_num, iteration=self._debug_iter)

            F_x_final, x = self.perform_newton_step_line_search(F_x, J_F, x)

            if torch.sum(F_x_final ** 2) < self.tol:
                break

            self._debug_iter += 1

        if torch.sum(F_x_final ** 2) > 0.1:
            print(f"Norm : {torch.sum(F_x_final ** 2)}, ({self._debug_iter})")

        self._debug_F_x.append(F_x_final)

        return x

    def simulate(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        dt: float,
    ) -> None:
        B = self.model.body_count  # Number of bodies
        C = self.model.max_contacts_per_body  # Max contacts per body
        D = self.model.joint_count  # Joint count

        self._compute_attributes(state_in, control, dt)

        # Initial integration without contacts
        _, body_vel = self.integrator.integrate(
            body_q=state_in.body_q,
            body_qd=state_in.body_qd,
            body_f=self._body_force,
            body_inv_mass=self.model.body_inv_mass,
            body_inv_inertia=self.model.body_inv_inertia,
            g_accel=self.model.g_accel,
            dt=self._dt,
        )

        # Initialize variables
        lambda_n = torch.full((B, C, 1), 0.00, device=self.device)  # [B, C, 1]
        lambda_t = torch.full((B, 2 * C, 1), 0.00, device=self.device)  # [B, 2C, 1]
        gamma = torch.full((B, C, 1), 0.00, device=self.device)  # [B, C, 1]
        lambda_j = torch.full((5 * D, 1), 0.00, device=self.device)  # [5D, 1]

        x0 = self.flatten_variables(body_vel, lambda_n, lambda_t, gamma, lambda_j) # [B(6 + 4C) + 5D, 1]
        x = self.newton(x0)

        body_vel, lambda_n, lambda_t, gamma, lambda_j = self.unflatten_variables(x)

        state_out.body_qd = body_vel
        state_out.time = state_in.time + dt

        # Semi-implicit Euler integration
        state_out.body_q[:, :3] = state_in.body_q[:, :3] + body_vel[:, 3:] * dt
        state_out.body_q[:, 3:] = integrate_quat_exact_batch(
            state_in.body_q[:, 3:], body_vel[:, :3], dt
        )

    def simulate_xitorch(
            self,
            state_in: State,
            state_out: State,
            control: Control,
            dt: float,
    ) -> None:
        B = self.model.body_count  # Number of bodies
        C = self.model.max_contacts_per_body  # Max contacts per body
        D = self.model.joint_count  # Joint count

        self._compute_attributes(state_in, control, dt)

        # Initial integration without contacts
        _, body_vel = self.integrator.integrate(
            body_q=state_in.body_q,
            body_qd=state_in.body_qd,
            body_f=self._body_force,
            body_inv_mass=self.model.body_inv_mass,
            body_inv_inertia=self.model.body_inv_inertia,
            g_accel=self.model.g_accel,
            dt=self._dt,
        )

        # Initialize variables
        lambda_n = torch.full((B, C, 1), 0.00, device=self.device)  # [B, C, 1]
        lambda_t = torch.full((B, 2 * C, 1), 0.00, device=self.device)  # [B, 2C, 1]
        gamma = torch.full((B, C, 1), 0.00, device=self.device)  # [B, C, 1]
        lambda_j = torch.full((5 * D, 1), 0.00, device=self.device)  # [5D, 1]

        x0 = self.flatten_variables(body_vel, lambda_n, lambda_t, gamma, lambda_j) # [B(6 + 4C) + 5D, 1]

        def residual_wrapper(x,
                             body_trans,
                             body_vel_prev,
                             body_force,
                             J_n,
                             J_t,
                             J_j_p,
                             J_j_c,
                             penetration_depth,
                             contact_weight,
                             restitution,
                             dynamic_friction):
            return self.residual(x,
                                body_trans=body_trans,
                                body_vel_prev=body_vel_prev,
                                body_force=body_force,
                                J_n=J_n,
                                J_t=J_t,
                                J_j_p=J_j_p,
                                J_j_c=J_j_c,
                                penetration_depth=penetration_depth,
                                contact_weight=contact_weight,
                                restitution=restitution,
                                dynamic_friction=dynamic_friction,
                                )

        def newton_wrapper(fnc, x0, params, **config):
            return self.newton(x0)

        x = rootfinder(
            residual_wrapper,
            x0,
            params=(self._body_trans, self._body_vel_prev, self._body_force,
                    self._J_n, self._J_t, self._J_j_p, self._J_j_c,
                    self._penetration_depth, self._contact_weight, self.model.restitution,
                    self.model.dynamic_friction),
            method=newton_wrapper,
            maxiter=150,
        )

        body_vel, lambda_n, lambda_t, gamma, lambda_j = self.unflatten_variables(x)

        state_out.body_qd = body_vel
        state_out.time = state_in.time + dt

        # Semi-implicit Euler integration
        state_out.body_q[:, :3] = state_in.body_q[:, :3] + body_vel[:, 3:] * dt
        state_out.body_q[:, 3:] = integrate_quat_exact_batch(
            state_in.body_q[:, 3:], body_vel[:, :3], dt
        )


