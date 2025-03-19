import torch
import numpy as np
from typing import Tuple
from pbd_torch.model import Model, State, Control
from pbd_torch.transform import transform, quat_mul, normalize_quat, rotate_vectors


class NewtonContactSolver:
    def __init__(self, max_iterations=20, tolerance=1e-6):
        """Initialize the non-linear Newton contact solver.

        Args:
            max_iterations: Maximum number of Newton iterations
            tolerance: Convergence tolerance
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def fb_function(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fischer-Burmeister NCP function: a + b - sqrt(a^2 + b^2)

        Args:
            a: First parameter (typically distance or velocity)
            b: Second parameter (typically force)

        Returns:
            Value of the Fischer-Burmeister function
        """
        eps = 1e-10  # Small value to prevent numerical issues
        return a + b - torch.sqrt(a ** 2 + b ** 2 + eps)

    def fb_derivative_a(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Partial derivative of Fischer-Burmeister function with respect to a

        Args:
            a: First parameter
            b: Second parameter

        Returns:
            Partial derivative ∂ψ/∂a
        """
        eps = 1e-10
        norm = torch.sqrt(a ** 2 + b ** 2 + eps)

        # Handle the non-smooth point (a,b)=(0,0)
        if torch.all(norm < 1e-6):
            return torch.ones_like(a) * 0.0

        return 1.0 - a / norm

    def fb_derivative_b(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Partial derivative of Fischer-Burmeister function with respect to b

        Args:
            a: First parameter
            b: Second parameter

        Returns:
            Partial derivative ∂ψ/∂b
        """
        eps = 1e-10
        norm = torch.sqrt(a ** 2 + b ** 2 + eps)

        # Handle the non-smooth point (a,b)=(0,0)
        if torch.all(norm < 1e-6):
            return torch.ones_like(b)

        return 1.0 - b / norm


    def solve_newton_contact(self, model: Model, state: State, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solve contact constraints using non-smooth Newton's method.

        Args:
            model: Physics model
            state: Current state
            dt: Time step

        Returns:
            Tuple of updated velocities and Lagrange multipliers
        """
        # If no contacts, return current velocities
        if model.contact_count == 0:
            return state.body_qd, torch.zeros(0, device=model.device)

        # Extract parameters
        contact_body = model.contact_body
        contact_point = model.contact_point
        contact_normal = model.contact_normal

        # Number of bodies and contacts
        n_bodies = model.body_count
        n_contacts = model.contact_count

        # Initialize velocities and Lagrange multipliers
        v = state.body_qd.clone().view(-1)  # Flatten the velocities
        lambda_n = torch.zeros(n_contacts, device=model.device)

        # Initialize arrays for Jacobian and constraints
        J = torch.zeros((n_contacts, v.shape[0]), device=model.device)
        constraint_values = torch.zeros(n_contacts, device=model.device)

        # Compute mass matrix (block diagonal)
        M = torch.zeros((v.shape[0], v.shape[0]), device=model.device)
        for b in range(n_bodies):
            # Fill diagonal blocks with mass and inertia
            start_idx = b * 6
            M[start_idx:start_idx + 3, start_idx:start_idx + 3] = torch.inverse(model.body_inv_inertia[b])
            M[start_idx + 3:start_idx + 6, start_idx + 3:start_idx + 6] = torch.diag(
                torch.ones(3, device=model.device) / model.body_inv_mass[b])

        # Compute initial external forces
        f_ext = torch.zeros_like(v)
        # for b in range(n_bodies):
        #     start_idx = b * 6
        #     f_ext[start_idx:start_idx + 3] = state.body_f[b, :3]  # torques
        #     f_ext[start_idx + 3:start_idx + 6] = state.body_f[b, 3:] + model.gravity * model.body_mass[b]  # forces

        for b in range(n_bodies):
            start_idx = b * 6
            f_ext[start_idx + 3:start_idx + 6] = model.gravity * model.body_mass[b]

        # Setup the Newton iteration
        for newton_iter in range(self.max_iterations):
            # Compute constraint Jacobian and values for each contact
            for c in range(n_contacts):
                b = contact_body[c].item()
                r = contact_point[c]  # contact point in body frame
                n = -contact_normal[c]  # contact normal

                q = state.body_q[b, 3:]  # Orientation of the body

                # Get body velocity components
                start_idx = b * 6
                w = v[start_idx:start_idx + 3]
                v_linear = v[start_idx + 3:start_idx + 6]

                # Compute relative velocity at contact point
                v_rel = v_linear + rotate_vectors(torch.linalg.cross(w, r), q)

                # Compute normal velocity (constraint value)
                v_n = torch.dot(v_rel, n)

                # Store constraint value (for Fischer-Burmeister function)
                constraint_values[c] = v_n

                # Compute Jacobian row for this contact
                J_row = torch.zeros_like(v)

                # Angular part: J_w = (r × n)
                J_row[start_idx:start_idx + 3] = torch.cross(r, n)

                # Linear part: J_v = n
                J_row[start_idx + 3:start_idx + 6] = n

                # Store in Jacobian matrix
                J[c] = J_row

            # Compute FB function values
            fb_values = self.fb_function(constraint_values, lambda_n)

            # Compute residual vector
            residual_v = M @ v - dt * f_ext - J.transpose(0, 1) @ lambda_n
            residual_lambda = fb_values

            # Check convergence
            res_norm = torch.norm(torch.cat([residual_v, residual_lambda]))
            if res_norm < self.tolerance:
                break

            # Compute Jacobian matrix for the system
            # [ M   -J^T ] [ Δv      ] = [ -residual_v ]
            # [ J    D_λ    ] [ Δλ      ] = [ -residual_λ ]

            # Compute D_λ diagonal matrix (derivatives of FB function)
            D_lambda = torch.diag(self.fb_derivative_b(constraint_values, lambda_n))

            # Assemble KKT matrix
            n_vars = v.shape[0] + lambda_n.shape[0]
            KKT = torch.zeros((n_vars, n_vars), device=model.device)
            KKT[:v.shape[0], :v.shape[0]] = M
            KKT[:v.shape[0], v.shape[0]:] = -J.transpose(0, 1)
            KKT[v.shape[0]:, :v.shape[0]] = J
            KKT[v.shape[0]:, v.shape[0]:] = D_lambda

            # Assemble right-hand side
            rhs = torch.cat([-residual_v, -residual_lambda])

            # Solve the system (could use a more efficient solver here)
            try:
                delta = torch.linalg.solve(KKT, rhs)
            except:
                # Fallback to least-squares if direct solve fails
                delta, _ = torch.linalg.lstsq(KKT, rhs)

            # Extract updates
            delta_v = delta[:v.shape[0]]
            delta_lambda = delta[v.shape[0]:]

            # Apply updates with line search for robustness
            alpha = 1.0
            v = v + alpha * delta_v
            lambda_n = torch.max(lambda_n + alpha * delta_lambda, torch.zeros_like(lambda_n))

        # Reshape velocities back to original format
        updated_body_qd = v.view(n_bodies, 6)

        return updated_body_qd, lambda_n

    def simulate(self, model: Model, state_in: State, state_out: State, control: Control, dt: float):
        """Performs one simulation step using the non-smooth Newton method for contacts.

        Args:
            model: The physics model
            state_in: The current state
            state_out: The output state to be computed
            control: The control inputs
            dt: Time step
        """
        # Get initial state
        body_q = state_in.body_q.clone()
        body_qd = state_in.body_qd.clone()
        body_f = state_in.body_f.clone()

        # Save the contact points to the input state
        state_in.contact_count = model.contact_count
        state_in.contact_body = model.contact_body.clone()
        state_in.contact_point = model.contact_point.clone()
        state_in.contact_normal = model.contact_normal.clone()
        state_in.contact_point_idx = model.contact_point_idx.clone()
        state_in.contact_point_ground = model.contact_point_ground.clone()

        # 1. Apply semi-implicit Euler integration for unconstrained step
        for b in range(model.body_count):
            # Extract components
            x = body_q[b, :3].clone()  # Position
            q = body_q[b, 3:].clone()  # Rotation
            v = body_qd[b, 3:].clone()  # Linear velocity
            w = body_qd[b, :3].clone()  # Angular velocity
            t = body_f[b, :3].clone()  # Torque
            f = body_f[b, 3:].clone()  # Linear force
            c = torch.linalg.cross(w, model.body_inv_inertia[b] @ w)  # Coriolis force

            # Update velocities with forces
            v = v + (f * model.body_inv_mass[b] + model.gravity) * dt
            w = w + model.body_inv_inertia[b] @ (t - c) * dt

            # Store updated velocities
            body_qd[b, :3] = w
            body_qd[b, 3:] = v

        # 2. Solve contact constraints using Newton's method
        if model.contact_count > 0:
            # Create a temporary state with updated velocities
            temp_state = State()
            temp_state.body_q = body_q
            temp_state.body_qd = body_qd

            # Solve constraints
            body_qd, lambda_n = self.solve_newton_contact(model, temp_state, dt)

        # 3. Update positions using corrected velocities
        for b in range(model.body_count):
            x = body_q[b, :3]
            q = body_q[b, 3:]
            v = body_qd[b, 3:]
            w = body_qd[b, :3]

            # Update position
            new_x = x + v * dt

            # Update orientation using quaternion integration
            w_q = torch.cat([torch.tensor([0.0], device=w.device), w])
            delta_q = 0.5 * quat_mul(w_q, q) * dt
            new_q = q + delta_q
            new_q = normalize_quat(new_q)

            body_q[b, :3] = new_x
            body_q[b, 3:] = new_q

        # Save the final state
        state_out.body_q = body_q
        state_out.body_qd = body_qd
        state_out.time = state_in.time + dt


class NonlinearNCGContactSolver:
    """Implementation of the Nonsmooth Nonlinear Conjugate Gradient method for solving contact problems."""

    def __init__(self, max_iterations=20, tolerance=1e-6, pgs_iterations=5):
        """Initialize the NNCG contact solver.

        Args:
            max_iterations: Maximum number of NNCG iterations
            tolerance: Convergence tolerance
            pgs_iterations: Number of PGS iterations for each sweep
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.pgs_iterations = pgs_iterations

    def compute_mass_matrix(self, model: Model) -> torch.Tensor:
        """Computes the global mass matrix.

        Args:
            model: Physics model

        Returns:
            Block diagonal mass matrix
        """
        n_bodies = model.body_count
        dim = n_bodies * 6
        M = torch.zeros((dim, dim), device=model.device)

        for b in range(n_bodies):
            # Fill diagonal blocks with mass and inertia
            idx = b * 6
            M[idx:idx + 3, idx:idx + 3] = torch.inverse(model.body_inv_inertia[b])
            M[idx + 3:idx + 6, idx + 3:idx + 6] = torch.diag(
                torch.ones(3, device=model.device) / model.body_inv_mass[b])

        return M

    def compute_jacobian(self, model: Model, state: State) -> torch.Tensor:
        """Computes the contact Jacobian matrix.

        Args:
            model: Physics model
            state: Current state

        Returns:
            Contact Jacobian matrix
        """
        n_bodies = model.body_count
        n_contacts = model.contact_count
        dim = n_bodies * 6

        # Initialize Jacobian
        J = torch.zeros((n_contacts, dim), device=model.device)

        for c in range(n_contacts):
            b = model.contact_body[c].item()
            r = model.contact_point[c]  # contact point in body frame
            n = -model.contact_normal[c]  # contact normal

            # Body index in the system
            idx = b * 6

            # Angular part: J_w = (r × n)
            J[c, idx:idx + 3] = torch.cross(r, n)

            # Linear part: J_v = n
            J[c, idx + 3:idx + 6] = n

        return J

    def pgs_solver(self, A: torch.Tensor, b: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Projected Gauss-Seidel solver for LCP.

        Args:
            A: System matrix
            b: Right-hand side
            x0: Initial guess

        Returns:
            Solution after PGS iterations
        """
        x = x0.clone()

        for _ in range(self.pgs_iterations):
            for i in range(len(b)):
                old_x_i = x[i].clone()

                # Compute residual for row i
                sum_ax = torch.dot(A[i, :], x)
                delta = (sum_ax - b[i]) / A[i, i] if A[i, i] > 1e-10 else 0.0

                # Update with projection onto non-negative orthant
                x[i] = torch.max(torch.tensor(0.0, device=x.device), x[i] - delta)

        return x

    def nonsmooth_nonlinear_cg(self, A: torch.Tensor, b: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Implements the Nonsmooth Nonlinear Conjugate Gradient method from Algorithm 10.

        Args:
            A: Constraint matrix
            b: Right-hand side vector
            x0: Initial guess

        Returns:
            Solution to the LCP
        """
        x = x0.clone()

        # Initial PGS sweep
        x1 = self.pgs_solver(A, b, x)

        # Initial gradient and direction
        grad_f0 = -(x1 - x)
        p0 = -grad_f0

        x = x1
        p = p0

        for k in range(self.max_iterations):
            # Next PGS sweep
            x_next = self.pgs_solver(A, b, x)

            # Compute gradient
            grad_f = -(x_next - x)

            # Compute beta (Fletcher-Reeves formula)
            grad_f_norm_squared = torch.sum(grad_f ** 2)
            prev_grad_f_norm_squared = torch.sum(grad_f0 ** 2)

            if prev_grad_f_norm_squared < 1e-10:
                beta = 0.0
            else:
                beta = grad_f_norm_squared / prev_grad_f_norm_squared

            # Check for restart
            if beta > 1.0:
                p = torch.zeros_like(p)  # restart
            else:
                # Update solution and search direction
                x_next = x_next + beta * p
                p = beta * p - grad_f

            # Check convergence
            if torch.norm(grad_f) < self.tolerance:
                return x_next

            # Update for next iteration
            x = x_next
            grad_f0 = grad_f

        return x

    def solve_contact_constraints(self, model: Model, state: State, dt: float) -> torch.Tensor:
        """Solve contact constraints using NNCG method.

        Args:
            model: Physics model
            state: Current state
            dt: Time step

        Returns:
            Updated body velocities
        """
        if model.contact_count == 0:
            return state.body_qd

        # Compute mass matrix and Jacobian
        M = self.compute_mass_matrix(model)
        J = self.compute_jacobian(model, state)

        # Prepare velocities as a flattened vector
        v = state.body_qd.clone().reshape(-1)

        # Compute Delassus operator: A = J M^-1 J^T
        J_M_inv = J @ torch.inverse(M)
        A = J_M_inv @ J.transpose(0, 1)

        # Compute right-hand side: b = -J v - bias
        b = -J @ v

        # Initialize impulses
        lambda_n = torch.zeros(model.contact_count, device=model.device)

        # Solve LCP using NNCG: A λ >= b, λ >= 0, λ^T (A λ - b) = 0
        lambda_n = self.nonsmooth_nonlinear_cg(A, b, lambda_n)

        # Compute velocity update: v += M^-1 J^T λ
        delta_v = torch.inverse(M) @ (J.transpose(0, 1) @ lambda_n)
        updated_v = v + delta_v

        # Reshape back to body_qd format
        body_qd = updated_v.reshape(model.body_count, 6)

        return body_qd

    def simulate(self, model: Model, state_in: State, state_out: State, control: Control, dt: float):
        """Performs one simulation step using NNCG for contacts.

        Args:
            model: The physics model
            state_in: The current state
            state_out: The output state to be computed
            control: The control inputs
            dt: Time step
        """
        # Get initial state
        body_q = state_in.body_q.clone()
        body_qd = state_in.body_qd.clone()
        body_f = state_in.body_f.clone()

        # Save the contact points to the input state
        state_in.contact_count = model.contact_count
        state_in.contact_body = model.contact_body.clone()
        state_in.contact_point = model.contact_point.clone()
        state_in.contact_normal = model.contact_normal.clone()
        state_in.contact_point_idx = model.contact_point_idx.clone()
        state_in.contact_point_ground = model.contact_point_ground.clone()

        # 1. Apply semi-implicit Euler integration for unconstrained step
        for b in range(model.body_count):
            # Extract components
            x = body_q[b, :3].clone()  # Position
            q = body_q[b, 3:].clone()  # Rotation
            v = body_qd[b, 3:].clone()  # Linear velocity
            w = body_qd[b, :3].clone()  # Angular velocity
            t = body_f[b, :3].clone()  # Torque
            f = body_f[b, 3:].clone()  # Linear force
            c = torch.linalg.cross(w, model.body_inv_inertia[b] @ w)  # Coriolis force

            # Update velocities with forces
            v = v + (f * model.body_inv_mass[b] + model.gravity) * dt
            w = w + model.body_inv_inertia[b] @ (t - c) * dt

            # Store updated velocities
            body_qd[b, :3] = w
            body_qd[b, 3:] = v

        # 2. Solve contact constraints using NNCG
        if model.contact_count > 0:
            # Create a temporary state with updated velocities
            temp_state = State()
            temp_state.body_q = body_q
            temp_state.body_qd = body_qd

            # Solve constraints
            body_qd = self.solve_contact_constraints(model, temp_state, dt)

        # 3. Update positions using corrected velocities
        for b in range(model.body_count):
            x = body_q[b, :3]
            q = body_q[b, 3:]
            v = body_qd[b, 3:]
            w = body_qd[b, :3]

            # Update position
            new_x = x + v * dt

            # Update orientation using quaternion integration
            w_q = torch.cat([torch.tensor([0.0], device=w.device), w])
            delta_q = 0.5 * quat_mul(w_q, q) * dt
            new_q = q + delta_q
            new_q = normalize_quat(new_q)

            body_q[b, :3] = new_x
            body_q[b, 3:] = new_q

        # Save the final state
        state_out.body_q = body_q
        state_out.body_qd = body_qd
        state_out.time = state_in.time + dt
