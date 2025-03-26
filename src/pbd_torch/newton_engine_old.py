import torch
from pbd_torch.constraints import compute_contact_jacobian
from pbd_torch.constraints import ground_penetration
from pbd_torch.integrator import SemiImplicitEulerIntegrator
from pbd_torch.model import Control
from pbd_torch.model import Model
from pbd_torch.model import State
from pbd_torch.ncp import FisherBurmeister
from pbd_torch.transform import transform_points_batch

BETA = 0.5  # Damping coefficient for Baumgarte stabilization


def get_penetration_depth(
    body_q: torch.Tensor,
    contact_bodies_a: torch.Tensor,
    contact_bodies_b: torch.Tensor,
    contact_points_a: torch.Tensor,
    contact_points_b: torch.Tensor,
):
    # Get the distanct between the two contact points in the world frame
    contact_points_a_world = transform_points_batch(
        contact_points_a, body_q[contact_bodies_a]
    )
    contact_points_b_world = transform_points_batch(
        contact_points_b, body_q[contact_bodies_b]
    )

    return torch.norm(contact_points_a_world - contact_points_b_world, dim=2)


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
        v_n_prev: torch.Tensor,
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
            Tuple (res_1, res_2): Dynamics residual and penetration constraint residual.
        """
        body_count = model.body_count
        active_mask = state_in.contact_mask.unsqueeze(-1)  # [body_count, max_contacts, 1]
        inactive_mask = ~active_mask  # [body_count, max_contacts, 1]

        # Compute acceleration from velocity change
        a = (body_qd - state_in.body_qd) / dt  # [body_count, 6, 1]

        # Dynamics residual: M @ a - J_n^T @ lambda_n - f = 0
        res_1 = (
            torch.bmm(model.mass_matrix, a)
            - torch.bmm(J_n.transpose(1, 2), lambda_n) / dt
            - state_in.body_f
            - torch.bmm(model.mass_matrix, model.gravity.expand(body_count, 6, 1))
        )  # [body_count, 8, 1]

        # Normal contact velocity
        v_n = torch.bmm(J_n, body_qd)  # [body_count, max_contacts, 1]
        penetration_depth = ground_penetration(
            state_in.body_q,
            state_in.contact_mask,
            state_in.contact_points,
            state_in.contact_points_ground,
            state_in.contact_normals,
        )  # [body_count, max_contacts, 1]

        # Penetration constraint residual using Fisher-Burmeister function
        error_bias = BETA * penetration_depth  # [body_count, max_contacts, 1]
        restitution_bias = model.restitution * v_n_prev  # [body_count, max_contacts, 1]
        penetration_bias = (
            error_bias + restitution_bias
        )  # [body_count, max_contacts, 1]
        res_2_active = self.fb.evaluate(
            lambda_n, v_n + penetration_bias
        )  # [body_count, max_contacts, 1]
        res_2_inactive = -lambda_n
        res_2 = active_mask * res_2_active + inactive_mask * res_2_inactive
        # res_2 = torch.where(state_in.contact_mask.unsqueeze(-1), res_2_active, res_2_inactive)


        # if state_in.time >= 1.65:
        #     print(f"Time: {state_in.time} Residuals: {res_1.norm()}, {res_2.norm()}")

        return res_1, res_2

    def compute_jacobian(
        self,
        model: Model,
        state_in: State,
        body_qd: torch.Tensor,
        lambda_n: torch.Tensor,
        dt: float,
        J_n: torch.Tensor,
        v_n_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble the Jacobian matrix for the Newton system.

        Args:
            model: Simulation model.
            state_in: Input state.
            body_qd: Current velocities [body_count, 6, 1].
            lambda_n: Contact forces [body_count, max_contacts].
            dt: Time step.
            J_n: Contact Jacobian [body_count, max_contacts, 6].

        Returns:
            J_F: Jacobian matrix [body_count, 6 + max_contacts, 6 + max_contacts].
        """
        # Contact velocity term
        v_n = torch.bmm(J_n, body_qd)  # [body_count, max_contacts, 1]

        active_mask = state_in.contact_mask.unsqueeze(-1)  # [body_count, max_contacts, 1]
        inactive_mask = ~active_mask  # [body_count, max_contacts, 1]

        # Fisher-Burmeister derivatives
        penetration_depth = ground_penetration(
            state_in.body_q,
            state_in.contact_mask,
            state_in.contact_points,
            state_in.contact_points_ground,
            state_in.contact_normals,
        )  # [body_count, max_contacts, 1]
        error_bias = BETA * penetration_depth  # [body_count, max_contacts, 1]
        restitution_bias = model.restitution * v_n_prev  # [body_count, max_contacts, 1]
        penetration_bias = (
            error_bias + restitution_bias
        )  # [body_count, max_contacts, 1]
        da_active, db = self.fb.derivatives(
            lambda_n, v_n + penetration_bias
        )  # [body_count, max_contacts, 1]

        da_inactive = -torch.ones_like(da_active)
        da = active_mask * da_active + inactive_mask * da_inactive

        # Jacobian blocks
        M = model.mass_matrix
        top_left = M / dt  # ∂res_1/∂body_qd
        top_right = -J_n.transpose(1, 2) / dt  # ∂res_1/∂lambda_n
        bottom_left = db * J_n  # ∂res_2/∂body_qd
        bottom_right = torch.diag_embed(da.squeeze(-1))  # ∂res_2/∂lambda_n

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
        delta_x = torch.linalg.solve(J_F, -F_x)
        new_body_qd = body_qd + delta_x[:, :6]
        new_lambda_n = lambda_n + delta_x[:, 6:]
        return new_body_qd, new_lambda_n

    def simulate(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        control: Control,
        dt: float,
    ):
        body_count = model.body_count
        max_contacts = state_in.contact_points.shape[1]

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
        lambda_n = torch.ones((body_count, max_contacts, 1), device=self.device)
        J_n = compute_contact_jacobian(state_in)  # [body_count, max_contacts, 6]
        v_n_prev = torch.bmm(J_n, state_in.body_qd)  # [body_count, max_contacts, 1]

        # Newton iteration loop
        for _ in range(self.iterations):
            res_1, res_2 = self.compute_residuals(
                model, state_in, new_body_qd, lambda_n, dt, J_n, v_n_prev
            )
            F_x = torch.cat((res_1, res_2), dim=1)
            J_F = self.compute_jacobian(
                model, state_in, new_body_qd, lambda_n, dt, J_n, v_n_prev
            )
            new_body_qd, lambda_n = self.perform_newton_step(
                J_F, F_x, new_body_qd, lambda_n
            )

        # Update output state
        state_out.body_q = new_body_q
        state_out.body_qd = new_body_qd
        state_out.time = state_in.time + dt