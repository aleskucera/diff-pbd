import os
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
from pbd_torch.transform import transform_multiply_batch
from pbd_torch.transform import rotate_vectors_batch


def forces_from_joint_actions(
        body_trans: torch.Tensor,
        joint_parent: torch.Tensor,
        joint_child: torch.Tensor,
        joint_trans_parent: torch.Tensor,
        joint_trans_child: torch.Tensor,
        joint_act: torch.Tensor,
) -> torch.Tensor:
    device = body_trans.device
    B = body_trans.shape[0]
    D = joint_parent.shape[0]

    body_f = torch.zeros((B, 6, 1), dtype=torch.float32, device=device)

    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device).repeat(D, 1).unsqueeze(-1) # [D, 3, 1]

    trans_parent = body_trans[joint_parent]  # [D, 7, 1]
    trans_child = body_trans[joint_child]  # [D, 7, 1]

    # Joint frame computed from the parent and child bodies
    X_p = transform_multiply_batch(trans_parent, joint_trans_parent)  # [D, 7, 1]
    X_c = transform_multiply_batch(trans_child, joint_trans_child) # [D, 7, 1]

    z_p = rotate_vectors_batch(z_axis, X_p[:, 3:]) # [D, 3, 1]
    z_c = rotate_vectors_batch(z_axis, X_c[:, 3:]) # [D, 3, 1]

    # TODO: Resolve the joint actions from joint_axis_mode (now just for forces)
    torque_p = z_p * joint_act.unsqueeze(1) # [D, 3, 1]
    torque_c = z_c * joint_act.unsqueeze(1) # [D, 3, 1]

    force_p = torch.zeros((D, 6, 1), dtype=torch.float32, device=device)
    force_p[:, :3] = -torque_p # [D, 3, 1]

    force_c = torch.zeros((D, 6, 1), dtype=torch.float32, device=device)
    force_c[:, :3] = torque_c # [D, 3, 1]

    body_f.index_add_(0, joint_parent, force_p)
    body_f.index_add_(0, joint_child, force_c)

    return body_f

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
        D = model.joint_parent.shape[0]  # Joint count

        self._J_n = torch.zeros((B, C, 6), device=device)  # [B, C, 6]
        self._J_t = torch.zeros((B, 2 * C, 6), device=device)  # [B, 2C, 6]
        self._J_j_p = torch.zeros((5 * D, 6), device=device)  # [5D, 6]
        self._J_j_c = torch.zeros((5 * D, 6), device=device)  # [5D, 6]
        self._body_trans = torch.zeros((B, 7, 1), device=device)  # [B, 7, 1]
        self._penetration_depth = torch.zeros((B, C, 1), device=device)  # [B, C, 1]
        self._contact_mask = torch.zeros(
            (B, C), device=device, dtype=torch.bool
        )  # [B, C]
        self._contact_mask_n = torch.zeros((B, C), device=device, dtype=torch.bool)  # [B, C]
        self._contact_mask_t = torch.zeros((B, C), device=device, dtype=torch.bool)  # [B, C]
        self._body_vel_prev = torch.zeros((B, 6, 1), device=device)  # [B, 6, 1]
        self._body_force = torch.zeros((B, 6, 1), device=device)  # [B, 6, 1]
        self._dt = None

        self._debug_iter = 0
        self._debug_F_x = []

    def residual(
        self,
        x: Float[torch.Tensor, "B(6 + 4C) + 5D 1"],
        dt: float
    ) -> Float[torch.Tensor, "B(6 + 4C) + 5D 1"]:
        """
        Compute the residual F(x) for root finding with XiTorch.

        Args:
            x: Flattened variables tensor [B(6 + 4C) + 5D, 1]
            dt: Time step (float)

        Returns:
            F_x: Flattened residual tensor [B(6 + 4C) + 5D, 1]
        """
        body_vel, lambda_n, lambda_t, gamma, lambda_j = self.unflatten_variables(x)
        return self.compute_F_x(body_vel, lambda_n, lambda_t, gamma, lambda_j, dt)

    def flatten_variables(
        self,
        body_vel: Float[torch.Tensor, "B 6 1"],
        lambda_n: Float[torch.Tensor, "B C 1"],
        lambda_t: Float[torch.Tensor, "B 2C 1"],
        gamma: Float[torch.Tensor, "B C 1"],
        lambda_j: Float[torch.Tensor, "5D 1"],
    ) -> Float[torch.Tensor, "B(6 + 4C) + 5D 1"]:
        """
        Flattens all variables into a single vector.

        Args:
            body_vel: Body velocities [B, 6, 1].
            lambda_n: Normal impulses [B, C, 1].
            lambda_t: Tangential impulses [B, 2C, 1].
            gamma: Friction auxiliary variables [B, C, 1].
            lambda_j: Joint impulses [5D, 1].

        Returns:
            Flattened vector [B(6 + 4C) + 5D, 1].
        """
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
        """
        Unflattens a vector back into individual variables.

        Args:
            x: Flattened vector [B(6 + 4C) + 5D, 1].

        Returns:
            Tuple of:
                body_vel [B, 6, 1],
                lambda_n [B, C, 1],
                lambda_t [B, 2C, 1],
                gamma [B, C, 1],
                lambda_j [5D, 1].
        """
        B = self.model.body_count
        C = self.model.max_contacts_per_body
        D = self.model.joint_parent.shape[0]
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

    def compute_F_x(
        self,
        body_vel: Float[torch.Tensor, "B 6 1"],
        lambda_n: Float[torch.Tensor, "B C 1"],
        lambda_t: Float[torch.Tensor, "B 2C 1"],
        gamma: Float[torch.Tensor, "B C 1"],
        lambda_j: Float[torch.Tensor, "5D 1"],
        dt: float,
    ) -> Float[torch.Tensor, "B(6 + 4C) + 5D 1"]:
        """
        Computes the flattened residuals vector F(x).

        Args:
            body_vel: Body velocities [B, 6, 1].
            lambda_n: Normal impulses [B, C, 1].
            lambda_t: Tangential impulses [B, 2C, 1].
            gamma: Friction auxiliary variables [B, C, 1].
            lambda_j: Joint impulses [5D, 1].
            dt: Time step.

        Returns:
            Flattened residuals vector F(x) [B(6 + 4C) + 5D, 1].
        """
        res_d, res_n, res_t, res_j = self.compute_residuals(
            body_vel, lambda_n, lambda_t, gamma, lambda_j, dt
        )
        F_x_batched = torch.cat((res_d, res_n, res_t), dim=1)  # [B, 6 + C + 3C, 1]
        F_x = torch.cat((F_x_batched.view(-1, 1), res_j), dim=0)  # [B(6 + 4C) + 5D, 1]
        return F_x

    def compute_residuals(
        self,
        body_vel: Float[torch.Tensor, "B 6 1"],
        lambda_n: Float[torch.Tensor, "B C 1"],
        lambda_t: Float[torch.Tensor, "B 2C 1"],
        gamma: Float[torch.Tensor, "B C 1"],
        lambda_j: Float[torch.Tensor, "5D 1"],
        dt: float,
    ) -> Tuple[
        Float[torch.Tensor, "B 6 1"],
        Float[torch.Tensor, "B C 1"],
        Float[torch.Tensor, "B 3C 1"],
        Float[torch.Tensor, "5D 1"],
    ]:
        """
        Computes the residuals for dynamics, contact, friction, and joint constraints.

        Args:
            body_vel: Body velocities [B, 6, 1].
            lambda_n: Normal impulses [B, C, 1].
            lambda_t: Tangential impulses [B, 2C, 1].
            gamma: Friction auxiliary variables [B, C, 1].
            lambda_j: Joint impulses [5D, 1].
            dt: Time step.

        Returns:
            Tuple of:
                res_dynamics [B, 6, 1],
                res_contact [B, C, 1],
                res_friction [B, 3C, 1],
                res_joint [5D, 1].
        """

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
            self._J_j_p,
            self._J_j_c,
            dt,
        )  # [B, 6, 1]

        # ------------------ Normal contact residual ------------------
        res_contact = self.contact_constraint.get_residuals(
            body_vel,
            self._body_vel_prev,
            lambda_n,
            self._J_n,
            self._penetration_depth,
            self._contact_mask,
            self.model.restitution,
            dt,
        )  # [B, C, 1]

        # ------------------ Friction contact residuals ------------------
        res_friction = self.friction_constraint.get_residuals(
            body_vel,
            lambda_n,
            lambda_t,
            gamma,
            self._J_t,
            self._contact_mask,
            self.model.dynamic_friction,
        )  # [B, 3C, 1]

        # ---------------- Joint residuals ------------------
        res_joint = self.revolute_constraint.get_residuals(
            body_vel, lambda_j, self._body_trans, self._J_j_p, self._J_j_c, dt
        )  # [5D, 1]

        return res_dynamics, res_contact, res_friction, res_joint

    def compute_jacobian(
        self,
        body_vel: Float[torch.Tensor, "B 6 1"],
        lambda_n: Float[torch.Tensor, "B C 1"],
        lambda_t: Float[torch.Tensor, "B 2C 1"],
        gamma: Float[torch.Tensor, "B C 1"],
        lambda_j: Float[torch.Tensor, "5D 1"],
        dt: float,
    ) -> torch.Tensor:
        """
        Computes the Jacobian matrix for the residuals.

        Args:
            body_vel: Body velocities [B, 6, 1].
            lambda_n: Normal impulses [B, C, 1].
            lambda_t: Tangential impulses [B, 2C, 1].
            gamma: Friction auxiliary variables [B, C, 1].
            lambda_j: Joint impulses [5D, 1].
            dt: Time step.

        Returns:
            Sparse COO Jacobian matrix [B(6 + 4C) + 5D, B(6 + 4C) + 5D].
        """
        B = body_vel.shape[0]  # Number of bodies
        C = lambda_n.shape[1]  # Max contacts per body
        D = self.model.joint_parent.shape[0]  # Joint count
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
            self.model.restitution,
            dt,
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
            body_vel, self._body_trans, self._J_j_p, self._J_j_c, dt
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
        body_vel: Float[torch.Tensor, "B 6 1"],
        lambda_n: Float[torch.Tensor, "B C 1"],
        lambda_t: Float[torch.Tensor, "B 2C 1"],
        gamma: Float[torch.Tensor, "B C 1"],
        lambda_j: Float[torch.Tensor, "5D 1"],
        dt: float,
        eps: float = 1e-6,
    ) -> Float[torch.Tensor, "B(6 + 4C) + 5D B(6 + 4C) + 5D"]:
        """
        Computes the numerical Jacobian using finite differences.

        Args:
            body_vel: Body velocities [B, 6, 1].
            lambda_n: Normal impulses [B, C, 1].
            lambda_t: Tangential impulses [B, 2C, 1].
            gamma: Friction auxiliary variables [B, C, 1].
            lambda_j: Joint impulses [5D, 1].
            dt: Time step.
            eps: Perturbation size for finite differences.

        Returns:
            Numerical Jacobian matrix [B(6 + 4C) + 5D, B(6 + 4C) + 5D].
        """

        B = self.model.body_count  # Body count
        C = self.model.max_contacts_per_body  # Max contacts per body
        D = self.model.joint_parent.shape[0]  # Joint count
        total_dim_batched = 6 + 4 * C  # Body velocities + contact variables
        full_size = B * total_dim_batched + 5 * D  # Total variables including joints

        # Flatten the variables into a single vector
        x = self.flatten_variables(body_vel, lambda_n, lambda_t, gamma, lambda_j)
        F_x = self.compute_F_x(body_vel, lambda_n, lambda_t, gamma, lambda_j, dt)

        # Initialize numerical Jacobian
        J_num = torch.zeros((full_size, full_size), device=self.device)

        # Compute each column of the Jacobian
        for i in range(full_size):
            x_plus = x.clone()
            x_plus[i, 0] += eps  # Perturb the i-th variable
            body_vel_p, lambda_n_p, lambda_t_p, gamma_p, lambda_j_p = (
                self.unflatten_variables(x_plus)
            )
            F_x_plus = self.compute_F_x(
                body_vel_p, lambda_n_p, lambda_t_p, gamma_p, lambda_j_p, dt
            )
            J_num[:, i] = (F_x_plus - F_x).squeeze() / eps

        return J_num

    def visualize_jacobian_comparison(
        self,
        J_ana: torch.Tensor,
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
        J_F: torch.Tensor,
        F_x: Float[torch.Tensor, "B(6 + 4C) + 5D 1"],
        body_vel: Float[torch.Tensor, "B 6 1"],
        lambda_n: Float[torch.Tensor, "B C 1"],
        lambda_t: Float[torch.Tensor, "B 2C 1"],
        gamma: Float[torch.Tensor, "B C 1"],
        lambda_j: Float[torch.Tensor, "5D 1"],
        dt: float,
    ) -> Tuple[
        Float[torch.Tensor, "B(6 + 4C) + 5D 1"],
        Float[torch.Tensor, "B 6 1"],
        Float[torch.Tensor, "B C 1"],
        Float[torch.Tensor, "B 2C 1"],
        Float[torch.Tensor, "B C 1"],
        Float[torch.Tensor, "5D 1"],
    ]:
        """
        Performs a Newton step with line search to update variables.

        Args:
            J_F: Sparse COO Jacobian matrix [B(6 + 4C) + 5D, B(6 + 4C) + 5D].
            F_x: Concatenated residuals [B(6 + 4C) + 5D, 1].
            body_vel: Body velocities [B, 6, 1].
            lambda_n: Normal impulses [B, C, 1].
            lambda_t: Tangential impulses [B, 2C, 1].
            gamma: Friction auxiliary variables [B, C, 1].
            lambda_j: Joint impulses [5D, 1].
            dt: Time step.

        Returns:
            Tuple of updated residuals, body velocities, normal impulses,
            tangential impulses, gamma, and joint impulses.
        """
        B = body_vel.shape[0]  # Number of bodies
        C = lambda_n.shape[1]  # Max contacts per body
        D = (
            self.model.joint_parent.shape[0]
            if self.model.joint_parent is not None
            else 0
        )  # Number of joints
        total_dim_batched = 6 + 4 * C  # Per-body variables
        full_size = (
            B * total_dim_batched + 5 * D
        )  # Total size including unbatched lambda_j

        F_x_flat = F_x.view(full_size, 1)  # [B(6+4C)+5D, 1]

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
        delta_x_flat = torch.linalg.solve(J_F_reg_dense, -F_x_flat)  # [full_size, 1]

        # Clamp the detlta_x to avoid huge jumps
        delta_x_flat = torch.clamp(delta_x_flat, -100.0, 100.0)

        # Split delta_x into batched and unbatched parts
        delta_x_batched = delta_x_flat[: B * total_dim_batched].view(
            B, total_dim_batched, 1
        )  # [B, 6+4C, 1]
        delta_body_qd = delta_x_batched[:, :6, :]  # [B, 6, 1]
        delta_lambda_n = delta_x_batched[:, 6 : 6 + C, :]  # [B, C, 1]
        delta_lambda_t = delta_x_batched[:, 6 + C : 6 + 3 * C, :]  # [B, 2C, 1]
        delta_gamma = delta_x_batched[:, 6 + 3 * C :, :]  # [B, C, 1]
        delta_lambda_j = delta_x_flat[B * total_dim_batched :]  # [5D, 1]

        norm = torch.sum(F_x**2)  # Scalar
        alpha = 1.0
        min_alpha = 1e-4
        max_linesearch_iters = 10

        for _ in range(max_linesearch_iters):
            body_qd_trial = body_vel + alpha * delta_body_qd  # [B, 6, 1]
            lambda_n_trial = lambda_n + alpha * delta_lambda_n  # [B, C, 1]
            lambda_t_trial = lambda_t + alpha * delta_lambda_t  # [B, 2C, 1]
            gamma_trial = gamma + alpha * delta_gamma  # [B, C, 1]
            lambda_j_trial = lambda_j + alpha * delta_lambda_j  # [5D, 1]

            res_d_new, res_n_new, res_t_new, res_j_new = self.compute_residuals(
                body_qd_trial,
                lambda_n_trial,
                lambda_t_trial,
                gamma_trial,
                lambda_j_trial,
                dt,
            )  # [B, 6, 1], [B, C, 1], [B, 3C, 1], [5D, 1]
            F_x_new_batched = torch.cat(
                [res_d_new, res_n_new, res_t_new], dim=1
            )  # [B, 6+4C, 1]
            F_x_new = torch.cat(
                [F_x_new_batched.view(-1, 1), res_j_new], dim=0
            )  # [B(6+4C)+5D, 1]
            new_norm = torch.sum(F_x_new**2)  # Scalar

            if new_norm < norm:
                norm = new_norm
                F_x = F_x_new
                break
            alpha /= 2.0
            if alpha < min_alpha:
                print("Linesearch failed to reduce residual; using smallest step")
                break

        new_body_qd = body_vel + alpha * delta_body_qd  # [B, 6, 1]
        new_lambda_n = lambda_n + alpha * delta_lambda_n  # [B, C, 1]
        new_lambda_t = lambda_t + alpha * delta_lambda_t  # [B, 2C, 1]
        new_gamma = gamma + alpha * delta_gamma  # [B, C, 1]
        new_lambda_j = lambda_j + alpha * delta_lambda_j  # [5D, 1]

        return F_x, new_body_qd, new_lambda_n, new_lambda_t, new_gamma, new_lambda_j

    def simulate(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        dt: float,
    ) -> None:
        """
        Simulates the system for one time step using the non-smooth Newton method.

        Args:
            state_in: Input state containing initial body and contact information.
            state_out: Output state to store updated body and contact information.
            control: Control inputs for the simulation.
            dt: Time step.
        """
        B = self.model.body_count  # Number of bodies
        C = state_in.contact_points_per_body.shape[1]  # Max contacts per body
        D = self.model.joint_parent.shape[0]  # Joint count

        control_body_f = forces_from_joint_actions(state_in.body_q,
                                                    self.model.joint_parent,
                                                    self.model.joint_child,
                                                    self.model.joint_X_p,
                                                    self.model.joint_X_c,
                                                    control.joint_act)
        body_f = state_in.body_f + control_body_f

        # Initial integration without contacts
        _, body_vel = self.integrator.integrate(
            body_q=state_in.body_q,
            body_qd=state_in.body_qd,
            body_f=body_f,
            body_inv_mass=self.model.body_inv_mass,
            body_inv_inertia=self.model.body_inv_inertia,
            gravity=self.model.g_accel,
            dt=dt,
        )

        self._body_force = body_f # [B, 6, 1]
        self._body_trans = state_in.body_q.clone()  # [B, 7, 1]
        self._body_vel_prev = state_in.body_qd.clone() # [B, 6, 1]
        self._contact_mask = state_in.contact_mask_per_body.clone() # [B, C, 1]

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

        # Initialize variables
        lambda_n = torch.full((B, C, 1), 0.00, device=self.device)  # [B, C, 1]
        lambda_t = torch.full((B, 2 * C, 1), 0.00, device=self.device)  # [B, 2C, 1]
        gamma = torch.full((B, C, 1), 0.00, device=self.device)  # [B, C, 1]
        lambda_j = torch.full((5 * D, 1), 0.00, device=self.device)  # [5D, 1]
        total_dim_batched = 6 + 4 * C  # Per-body variables
        F_x_final = torch.zeros(
            (B * total_dim_batched + 5 * D, 1), device=self.device
        )  # [B(6+4C)+5D, 1]

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

            F_x_batched = torch.cat((res_d, res_n, res_t), dim=1)  # [B, 6+4C, 1]
            F_x = torch.cat((F_x_batched.view(-1, 1), res_j), dim=0)  # [B(6+4C)+5D, 1]
            J_F = self.compute_jacobian(
                body_vel,
                lambda_n,
                lambda_t,
                gamma,
                lambda_j,
                dt,
            )


            # Perform Jacobian check only in the first iteration
            if self.debug_jacobian and i == 0:
                J_num = self.compute_numerical_jacobian(
                    body_vel, lambda_n, lambda_t, gamma, lambda_j, dt
                )
                self.visualize_jacobian_comparison(J_F, J_num, iteration=self._debug_iter)

            F_x_final, body_vel, lambda_n, lambda_t, gamma, lambda_j = (
                self.perform_newton_step_line_search(
                    J_F,
                    F_x,
                    body_vel,
                    lambda_n,
                    lambda_t,
                    gamma,
                    lambda_j,
                    dt,
                )
            )

            if torch.sum(F_x_final**2) < self.tol:
                break

            self._debug_iter += 1

        if torch.sum(F_x_final**2) > 0.1:
            print(
                f"Norm : {torch.sum(F_x_final ** 2)}, Time: {state_in.time + dt} ({self._debug_iter})"
            )
        self._debug_F_x.append(F_x_final)

        state_out.body_qd = body_vel
        state_out.time = state_in.time + dt

        # Semi-implicit Euler integration
        state_out.body_q[:, :3] = state_in.body_q[:, :3] + body_vel[:, 3:] * dt
        state_out.body_q[:, 3:] = integrate_quat_exact_batch(
            state_in.body_q[:, 3:], body_vel[:, :3], dt
        )

    def simulate_xitorch(self, state_in: State, state_out: State, control: Control, dt: float) -> None:
        B = self.model.body_count
        C = state_in.contact_points_per_body.shape[1]
        D = self.model.joint_parent.shape[0]

        control_body_f = forces_from_joint_actions(state_in.body_q,
                                                   self.model.joint_parent,
                                                   self.model.joint_child,
                                                   self.model.joint_X_p,
                                                   self.model.joint_X_c,
                                                   control.joint_act)
        body_f = state_in.body_f + control_body_f

        # Initial integration
        _, body_vel = self.integrator.integrate(
            body_q=state_in.body_q,
            body_qd=state_in.body_qd,
            body_f=body_f,
            body_inv_mass=self.model.body_inv_mass,
            body_inv_inertia=self.model.body_inv_inertia,
            gravity=self.model.g_accel,
            dt=dt,
        )

        self._body_force = body_f # [B, 6, 1]
        self._body_trans = state_in.body_q.clone()  # [B, 7, 1]
        self._body_vel_prev = state_in.body_qd.clone() # [B, 6, 1]
        self._contact_mask = state_in.contact_mask_per_body.clone() # [B, C, 1]

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

        # Initial guess
        lambda_n = torch.zeros((B, C, 1), device=self.device)
        lambda_t = torch.zeros((B, 2 * C, 1), device=self.device)
        gamma = torch.zeros((B, C, 1), device=self.device)
        lambda_j = torch.zeros((5 * D, 1), device=self.device)
        x0 = self.flatten_variables(body_vel, lambda_n, lambda_t, gamma, lambda_j)

        residual_func = lambda x, y: self.residual(x, y)

        # Solve using rootfinder
        x_solution = rootfinder(
            residual_func,
            x0,
            params=(dt,),
            alpha=1e-3,
            method="broyden1",
            linesearch="armijo",
            maxiter=self.iterations,
            ftol=self.tol,
        )

        # Unflatten and update state
        body_vel, lambda_n, lambda_t, gamma, lambda_j = self.unflatten_variables(x_solution)
        state_out.body_qd = body_vel
        state_out.time = state_in.time + dt
        state_out.body_q[:, :3] = state_in.body_q[:, :3] + body_vel[:, 3:] * dt
        state_out.body_q[:, 3:] = integrate_quat_exact_batch(state_in.body_q[:, 3:], body_vel[:, :3], dt)
