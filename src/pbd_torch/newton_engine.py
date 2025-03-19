import torch
from pbd_torch.constraints import compute_contact_jacobian
from pbd_torch.integrator import SemiImplicitEulerIntegrator
from pbd_torch.model import Control
from pbd_torch.model import Model
from pbd_torch.model import State
from pbd_torch.ncp import FisherBurmeister

BETA = 0.1  # Damping coefficient for Baumgarte stabilization


class NonSmoothNewtonEngine:
    def __init__(
        self, iterations: int = 10, device: torch.device = torch.device("cpu")
    ):
        self.device = device
        self.iterations = iterations
        self.integrator = SemiImplicitEulerIntegrator(
            use_local_omega=True, device=device
        )
        self.fb = FisherBurmeister(epsilon=1e-6)
        self.friction_epsilon = 1e-6  # Regularization for friction term

    def compute_residuals(
        self,
        model: Model,
        state_in: State,
        body_qd: torch.Tensor,
        lambda_n: torch.Tensor,
        dt: float,
        J_n: torch.Tensor,
        v_pre: torch.Tensor,
        e: torch.Tensor,
    ) -> tuple:
        """Compute residuals for the Newton iteration.

        Args:
            model: Simulation model containing mass matrix and gravity.
            state_in: Input state with previous velocities and forces.
            body_qd: Current velocities being solved for [body_count, 6].
            lambda_n: Contact forces [body_count, max_contacts].
            dt: Time step.
            J_n: Contact Jacobian [body_count, max_contacts, 6].

        Returns:
            Tuple (res_1, res_2): Dynamics residual and contact constraint residual.
        """
        # Compute acceleration from velocity change
        a = (body_qd - state_in.body_qd) / dt  # [body_count, 6]

        # Dynamics residual: M*a - J_n^T * lambda_n - f = 0
        f = state_in.body_f.clone()
        f[:, 3:] += model.gravity.unsqueeze(0)  # Add gravity to linear forces
        res_1 = (
            torch.bmm(model.mass_matrix, a.unsqueeze(-1)).squeeze(-1)
            - torch.bmm(J_n.transpose(1, 2), lambda_n.unsqueeze(-1)).squeeze(-1)
            - f
        )

        # Contact velocity term
        v_n = torch.bmm(J_n, body_qd.unsqueeze(-1)).squeeze(
            -1
        )  # [body_count, max_contacts]
        penetration_depth = state_in.contact_points[
            :, :, 2
        ]  # [body_count, max_contacts]

        # Contact constraint residual using Fisher-Burmeister function
        res_2 = self.fb.evaluate(
            lambda_n, v_n + BETA * penetration_depth + e * v_pre
        )  # [body_count, max_contacts]

        return res_1, res_2

    def compute_jacobian(
        self,
        model: Model,
        state_in: State,
        body_qd: torch.Tensor,
        lambda_n: torch.Tensor,
        dt: float,
        J_n: torch.Tensor,
        v_pre: torch.Tensor,
        e: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble the Jacobian matrix for the Newton system.

        Args:
            model: Simulation model.
            state_in: Input state.
            body_qd: Current velocities [body_count, 6].
            lambda_n: Contact forces [body_count, max_contacts].
            dt: Time step.
            J_n: Contact Jacobian [body_count, max_contacts, 6].

        Returns:
            J_F: Jacobian matrix [body_count, 6 + max_contacts, 6 + max_contacts].
        """
        # Contact velocity term
        v_n = torch.bmm(J_n, body_qd.unsqueeze(-1)).squeeze(-1)

        # Fisher-Burmeister derivatives
        penetration_depth = state_in.contact_points[
            :, :, 2
        ]  # [body_count, max_contacts]
        da, db = self.fb.derivatives(
            lambda_n, v_n + BETA * penetration_depth + e * v_pre
        )  # [body_count, max_contacts]

        # Jacobian blocks
        M = model.mass_matrix
        top_left = M / dt  # ∂res_1/∂body_qd
        top_right = -J_n.transpose(1, 2)  # ∂res_1/∂lambda_n
        bottom_left = db.unsqueeze(-1) * J_n  # ∂res_2/∂body_qd
        bottom_right = torch.diag_embed(da)  # ∂res_2/∂lambda_n

        # Assemble full Jacobian
        top_row = torch.cat([top_left, top_right], dim=2)
        bottom_row = torch.cat([bottom_left, bottom_right], dim=2)
        J_F = torch.cat([top_row, bottom_row], dim=1)

        return J_F

    def perform_newton_step(
        self,
        J_F: torch.Tensor,
        F_x: torch.Tensor,
        body_qd: torch.Tensor,
        lambda_n: torch.Tensor,
    ) -> tuple:
        """Solve the Newton system and update variables.

        Args:
            J_F: Jacobian matrix.
            F_x: Residual vector.
            body_qd: Current velocities to update.
            lambda_n: Current contact forces to update.

        Returns:
            Tuple (body_qd, lambda_n): Updated velocities and contact forces.
        """
        delta_x = torch.linalg.solve(J_F, -F_x.unsqueeze(-1)).squeeze(-1)
        body_qd += delta_x[:, :6]
        lambda_n += delta_x[:, 6:]
        return body_qd, lambda_n

    def simulate(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        control: Control,
        dt: float,
    ):
        """Simulate one time step using the non-smooth Newton method.

        Args:
            model: Simulation model.
            state_in: Input state.
            state_out: Output state to update.
            control: Control inputs (unused currently).
            dt: Time step.
        """
        body_count = model.body_count
        max_contacts = (
            state_in.contact_points.shape[1]
            if state_in.contact_points.numel() > 0
            else 0
        )

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
        lambda_n = torch.zeros((body_count, max_contacts), device=self.device)
        J_n = compute_contact_jacobian(
            state_in
        )  # Compute once, as it depends on state_in
        v_pre = torch.bmm(J_n, state_in.body_qd.unsqueeze(-1)).squeeze(
            -1
        )  # [body_count, max_contacts]
        # Restitution coefficient (broadcast if per-body)
        e = (
            model.restitution[:, None]
            if model.restitution.dim() == 1
            else model.restitution
        )

        # Newton iteration loop
        for _ in range(self.iterations):
            res_1, res_2 = self.compute_residuals(
                model, state_in, new_body_qd, lambda_n, dt, J_n, v_pre, e
            )
            F_x = torch.cat((res_1, res_2), dim=1)
            J_F = self.compute_jacobian(
                model, state_in, new_body_qd, lambda_n, dt, J_n, v_pre, e
            )
            new_body_qd, lambda_n = self.perform_newton_step(
                J_F, F_x, new_body_qd, lambda_n
            )

        # Update output state
        state_out.body_q = new_body_q
        state_out.body_qd = new_body_qd
        state_out.time = state_in.time + dt
