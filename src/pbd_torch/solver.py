import os
from typing import Tuple

import numpy as np
import torch
from pbd_torch.model import Control
from pbd_torch.model import Model
from pbd_torch.model import State
from pbd_torch.printer import DebugLogger
from pbd_torch.transform import *

os.environ["DEBUG"] = "false"


def external_forces(model: Model, state: State, control: Control):
    # Compute initial external forces
    f_ext = torch.zeros_like(model.body_qd.reshape(-1))
    for b in range(model.body_count):
        start_idx = b * 6
        f_ext[start_idx + 3 : start_idx + 6] = model.gravity * model.body_mass[b]


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """Compute the skew-symmetric matrix of a 3D vector.

    Args:
        v: 3D vector

    Returns:
        Skew-symmetric matrix
    """

    return torch.tensor([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def compute_contact_space(
    state: State,
    contact_body: torch.Tensor,
    contact_point: torch.Tensor,
    contact_normal: torch.Tensor,
):
    """Compute the contact space basis for each contact in vectorized vay. For Each contact the

    C = [n, t, b]

    where n is the contact normal and t, b are orthogonal vectors which are selected like this. Vector t is in the tangent velocity direction and the
    b is the cross product f the n and t.

    The C can also be split into the J_n and J_t which are the normal and tangent jacobians respectively.

    Args:
        state (State): The current state of the system
        contact_body (torch.Tensor): The body index for each contact of shape (n_contacts,)
        contact_point (torch.Tensor): The contact point in the body frame for each contact of shape (n_contacts, 3)
        contact_normal (torch.Tensor): The contact normal in the world frame for each contact of shape (n_contacts, 3)
    """
    # Compute the number of contacts
    n_contacts = contact_body.shape[0]

    J_n = torch.zeros((n_contacts, 6), device=contact_body.device)
    J_t = torch.zeros((n_contacts, 2, 6), device=contact_body.device)

    for c in range(n_contacts):
        b = contact_body[c]
        n = contact_normal[c]
        q = state.body_q[b, 3:]
        rx = skew_symmetric(rotate_vectors(contact_point[c], q))

        # Compute normal Jacobian row
        J_n_row = torch.zeros(6, device=contact_body.device)
        J_n_row[c : c + 3] = n @ rx  # Angular part
        J_n_row[c + 3 : c + 6] = -n  # Linear part
        J_n[c] = J_n_row

        # Compute the relative velocity at the contact point
        v_rel = state.body_qd[b, 3:] + rotate_vectors(
            torch.linalg.cross(state.body_qd[b, :3], contact_point[c]), q
        )

        v_n = torch.dot(v_rel, n)
        v_t = v_rel - v_n * n
        t = v_t / torch.norm(v_t)
        b = torch.cross(n, t)

        # Compute tangent Jacobian rows
        for t_idx, t_dir in enumerate([t, b]):
            J_t_row = torch.zeros(6, device=contact_body.device)
            J_t_row[c : c + 3] = t_dir @ rx
            J_t_row[c + 3 : c + 6] = -t_dir

        J_t[c] = J_t_row

    return J_n, J_t


class NewtonFrictionContactSolver:
    def __init__(self, max_iterations=5, tolerance=1e-6):
        """Initialize the non-linear Newton solver for contact and friction.

        Args:
            max_iterations: Maximum number of Newton iterations
            tolerance: Convergence tolerance
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def fb_function(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fischer-Burmeister NCP function: a + b - sqrt(a^2 + b^2)

        Args:
            a: First parameter
            b: Second parameter

        Returns:
            Value of the Fischer-Burmeister function
        """
        eps = 1e-10  # Small value to prevent numerical issues
        return a + b - torch.sqrt(a**2 + b**2 + eps)

    def fb_derivative_a(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Partial derivative of Fischer-Burmeister function with respect to a

        Args:
            a: First parameter
            b: Second parameter

        Returns:
            Partial derivative ∂ψ/∂a
        """
        eps = 1e-10
        norm = torch.sqrt(a**2 + b**2 + eps)

        # Handle the non-smooth point (a,b)=(0,0)
        mask = norm < 1e-6
        result = torch.ones_like(a)
        result[~mask] = 1.0 - a[~mask] / norm[~mask]
        result[mask] = 0.0  # Arbitrary choice at non-smooth point

        return result

    def fb_derivative_b(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Partial derivative of Fischer-Burmeister function with respect to b

        Args:
            a: First parameter
            b: Second parameter

        Returns:
            Partial derivative ∂ψ/∂b
        """
        eps = 1e-10
        norm = torch.sqrt(a**2 + b**2 + eps)

        # Handle the non-smooth point (a,b)=(0,0)
        mask = norm < 1e-6
        result = torch.ones_like(b)
        result[~mask] = 1.0 - b[~mask] / norm[~mask]
        result[mask] = 1.0  # Arbitrary choice at non-smooth point

        return result

    def compute_orthogonal_basis(
        self, normal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute two vectors orthogonal to the given normal vector.

        Args:
            normal: Unit normal vector

        Returns:
            Two orthogonal tangent vectors
        """
        # First find any vector not parallel to normal
        if torch.abs(normal[0]) < 0.9:
            v = torch.tensor([1.0, 0.0, 0.0], device=normal.device)
        else:
            v = torch.tensor([0.0, 1.0, 0.0], device=normal.device)

        # First tangent is cross product of normal and v
        t1 = torch.linalg.cross(normal, v)
        t1 = t1 / (torch.norm(t1) + 1e-10)

        # Second tangent is cross product of normal and first tangent
        t2 = torch.linalg.cross(normal, t1)

        return t1, t2

    def friction_derivative(self, lambda_t: torch.Tensor) -> torch.Tensor:
        """Compute the derivative of ||lambda_t|| with respect to lambda_t.

        Args:
            lambda_t: Friction force vector

        Returns:
            Derivative of norm with respect to lambda_t
        """
        norm = torch.norm(lambda_t)
        if norm < 1e-10:
            # Return unit vector in the direction of friction velocity
            return torch.zeros_like(lambda_t)
        else:
            return lambda_t / norm

    def solve_contact_friction(
        self, model: Model, state: State, dt: float, debug: DebugLogger
    ) -> torch.Tensor:
        """Solve contact and friction constraints using non-smooth Newton's method.

        Args:
            model: Physics model
            state: Current state
            dt: Time step
            debug: Debug printer

        Returns:
            Updated body velocities
        """
        debug.subsection("CONTACT FRICTION SOLVE")
        print("CONTACT FRICTION SOLVE")

        # If no contacts, return current velocities
        if model.contact_count == 0:
            debug.print("No contacts detected, skipping solver")
            return state.body_qd

        # Extract parameters
        contact_body = model.contact_body
        contact_point = model.contact_point
        contact_normal = model.contact_normal

        # Number of bodies and contacts
        n_bodies = model.body_count
        n_contacts = model.contact_count

        debug.print(f"Bodies: {n_bodies}, Contacts: {n_contacts}")

        # Initialize state variables
        # For each contact we have:
        # - 1 normal force (lambda_n)
        # - 2 tangential forces (lambda_t)
        # - 1 slack variable (gamma)
        v = state.body_qd.clone().reshape(-1)  # Flattened body velocities [6*n_bodies]
        lambda_n = torch.zeros(n_contacts, device=model.device)  # Normal forces
        lambda_t = torch.zeros(
            n_contacts, 2, device=model.device
        )  # Tangential forces (2D)
        gamma = torch.zeros(n_contacts, device=model.device)  # Slack variables

        debug.print("Initial variables:")
        debug.indent()
        debug.print(f"v shape: {v.shape}")
        debug.print(f"lambda_n shape: {lambda_n.shape}")
        debug.print(f"lambda_t shape: {lambda_t.shape}")
        debug.print(f"gamma shape: {gamma.shape}")
        debug.undent()

        # Pre-compute tangent basis for each contact
        tangent_basis = []
        debug.print("Computing tangent basis for each contact:")
        debug.indent()
        for c in range(n_contacts):
            n = -contact_normal[c]  # Contact normal (pointing from ground to body)
            t1, t2 = self.compute_orthogonal_basis(n)
            tangent_basis.append((t1, t2))
            debug.print(f"Contact {c} - Normal: {n}, Tangent1: {t1}, Tangent2: {t2}")
        debug.undent()

        # Initialize Jacobian matrices
        J_n = torch.zeros(
            (n_contacts, v.shape[0]), device=model.device
        )  # Normal Jacobian
        J_t = torch.zeros(
            (n_contacts, 2, v.shape[0]), device=model.device
        )  # Tangent Jacobian

        # Compute mass matrix (block diagonal)
        M = torch.zeros((v.shape[0], v.shape[0]), device=model.device)
        for b in range(n_bodies):
            idx = b * 6
            M[idx : idx + 3, idx : idx + 3] = torch.inverse(model.body_inv_inertia[b])
            M[idx + 3 : idx + 6, idx + 3 : idx + 6] = torch.diag(
                torch.ones(3, device=model.device) / model.body_inv_mass[b]
            )

        debug.print("Mass matrix constructed")

        # Compute initial external forces
        f_ext = torch.zeros_like(v)
        for b in range(n_bodies):
            start_idx = b * 6
            f_ext[start_idx + 3 : start_idx + 6] = model.gravity * model.body_mass[b]

        debug.print("Gravity forces applied")

        # Newton iteration
        debug.print("Starting Newton iterations:")
        for iter_idx in range(self.max_iterations):
            debug.print(f"Iteration {iter_idx + 1}/{self.max_iterations}")
            debug.indent()

            # Compute Jacobian and constraint values for each contact
            v_n = torch.zeros(n_contacts, device=model.device)  # Normal velocities
            v_t = torch.zeros(
                n_contacts, 2, device=model.device
            )  # Tangential velocities

            debug.print("Computing contact velocities and Jacobians:")
            debug.indent()
            for c in range(n_contacts):
                b = contact_body[c].item()
                r = contact_point[c]  # Contact point in body frame
                n = -contact_normal[c]  # Contact normal
                t1, t2 = tangent_basis[c]  # Tangent directions

                # Body position and rotation
                x = state.body_q[b, :3]
                q = state.body_q[b, 3:]

                # Get body velocity components
                idx = b * 6
                w = v[idx : idx + 3]  # Angular velocity
                v_linear = v[idx + 3 : idx + 6]  # Linear velocity

                # Compute relative velocity at contact point
                v_rel = v_linear + rotate_vectors(torch.linalg.cross(w, r), q)

                # Normal velocity
                v_n[c] = torch.dot(v_rel, n)

                # Tangential velocities
                v_t[c, 0] = torch.dot(v_rel, t1)
                v_t[c, 1] = torch.dot(v_rel, t2)

                debug.print(f"Contact {c} (Body {b}):")
                debug.indent()
                debug.print(f"Normal velocity: {v_n[c]:.6f}")
                debug.print(
                    f"Tangential velocities: [{v_t[c, 0]:.6f}, {v_t[c, 1]:.6f}]"
                )
                debug.undent()

                # Compute normal Jacobian row
                J_n_row = torch.zeros_like(v)
                J_n_row[idx : idx + 3] = torch.linalg.cross(r, n)  # Angular part
                J_n_row[idx + 3 : idx + 6] = n  # Linear part
                J_n[c] = J_n_row

                # Compute tangent Jacobian rows
                for t_idx, t_dir in enumerate([t1, t2]):
                    J_t_row = torch.zeros_like(v)
                    J_t_row[idx : idx + 3] = torch.linalg.cross(
                        r, t_dir
                    )  # Angular part
                    J_t_row[idx + 3 : idx + 6] = t_dir  # Linear part
                    J_t[c, t_idx] = J_t_row
            debug.undent()

            # Compute constraint violations
            debug.print("Computing constraint violations:")
            debug.indent()

            # 1. Normal constraint: Fischer-Burmeister function for 0 <= v_n ⊥ lambda_n >= 0
            fb_n = self.fb_function(v_n, lambda_n)
            debug.print(f"Normal constraint violations: {fb_n}")

            # 2. Friction constraints:
            # a) Friction dynamics: s(u, lambda_t, gamma) = J_t u + gamma * dλt
            s = torch.zeros(n_contacts, 2, device=model.device)
            for c in range(n_contacts):
                # For each tangent direction
                for t_idx in range(2):
                    # J_t * u: tangential velocity
                    s[c, t_idx] = v_t[c, t_idx]

                # Add friction direction term: gamma * ∂||λ_t||/∂λ_t
                lambda_t_norm = torch.norm(lambda_t[c])
                if lambda_t_norm > 1e-10:
                    s[c, 0] += gamma[c] * lambda_t[c, 0] / lambda_t_norm
                    s[c, 1] += gamma[c] * lambda_t[c, 1] / lambda_t_norm

            debug.print(f"Friction dynamics violations: {s}")

            # b) Complementarity: 0 <= gamma ⊥ mu*lambda_n - ||lambda_t|| >= 0
            # Compute the norm of tangential forces
            lambda_t_norms = torch.norm(lambda_t, dim=1)
            # Compute friction bound
            friction_bound = model.dynamic_friction[contact_body] * lambda_n
            # Compute complementarity function
            fb_t = self.fb_function(gamma, friction_bound - lambda_t_norms)
            debug.print(f"Friction cone violations: {fb_t}")
            debug.undent()

            # Compute residuals
            debug.print("Computing residual vector")
            residual_v = M @ v - dt * f_ext
            for c in range(n_contacts):
                # Add normal force contribution
                residual_v -= dt * J_n[c] * lambda_n[c]
                # Add tangential force contributions
                for t_idx in range(2):
                    residual_v -= dt * J_t[c, t_idx] * lambda_t[c, t_idx]

            # Flatten other residuals for easier concatenation
            residual_n = fb_n
            residual_s = s.reshape(-1)
            residual_t = fb_t

            # Concatenate all residuals
            residual = torch.cat([residual_v, residual_n, residual_s, residual_t])

            # Check convergence
            res_norm = torch.norm(residual)
            debug.print(f"Residual norm: {res_norm:.8f}")
            print(f"Residual norm: {res_norm:.8f}")
            if res_norm < self.tolerance:
                debug.print("Converged!")
                break

            # Build the Jacobian matrix for Newton method
            debug.print("Building Newton system Jacobian")

            # System size calculation:
            # - residual_v: 6*n_bodies equations
            # - residual_n: n_contacts equations
            # - residual_s: 2*n_contacts equations
            # - residual_t: n_contacts equations
            # Total: 6*n_bodies + 4*n_contacts

            n_vars = 6 * n_bodies + 4 * n_contacts
            J_newton = torch.zeros((n_vars, n_vars), device=model.device)

            # Partial derivatives for body velocities
            v_start = 0
            v_end = 6 * n_bodies

            # Partial derivatives for normal forces
            lambda_n_start = v_end
            lambda_n_end = lambda_n_start + n_contacts

            # Partial derivatives for tangential forces
            lambda_t_start = lambda_n_end
            lambda_t_end = lambda_t_start + 2 * n_contacts

            # Partial derivatives for slack variables
            gamma_start = lambda_t_end
            gamma_end = gamma_start + n_contacts

            # 1. Derivative of residual_v with respect to velocities
            J_newton[v_start:v_end, v_start:v_end] = M

            # 2. Derivative of residual_v with respect to forces
            for c in range(n_contacts):
                # w.r.t normal force
                J_newton[v_start:v_end, lambda_n_start + c] = -J_n[c]

                # w.r.t tangential forces
                for t_idx in range(2):
                    J_newton[v_start:v_end, lambda_t_start + 2 * c + t_idx] = -J_t[
                        c, t_idx
                    ]

            # 3. Derivative of normal constraint w.r.t velocities and normal forces
            for c in range(n_contacts):
                # ∂fb_n/∂v: Derivative through J_n
                dfb_dv = self.fb_derivative_a(v_n[c], lambda_n[c])
                J_newton[lambda_n_start + c, v_start:v_end] = J_n[c]

                # ∂fb_n/∂lambda_n: Direct derivative
                dfb_dlambda_n = self.fb_derivative_b(v_n[c], lambda_n[c])
                J_newton[lambda_n_start + c, lambda_n_start + c] = dfb_dlambda_n

            # 4. Derivative of friction dynamics w.r.t velocities, tangential forces, and gamma
            for c in range(n_contacts):
                for t_idx in range(2):
                    row_idx = lambda_n_end + 2 * c + t_idx
                    # ∂s/∂v
                    J_newton[row_idx, v_start:v_end] = J_t[c, t_idx]
                    # ∂s/∂lambda_t
                    lambda_t_norm = torch.norm(lambda_t[c]) + 1e-10
                    if gamma[c] > 1e-10 and lambda_t_norm > 1e-10:
                        for j in range(2):
                            if j == t_idx:
                                deriv = gamma[c] * (
                                    1 / lambda_t_norm
                                    - (lambda_t[c, t_idx] ** 2) / (lambda_t_norm**3)
                                )
                            else:
                                deriv = (
                                    -gamma[c]
                                    * lambda_t[c, t_idx]
                                    * lambda_t[c, j]
                                    / (lambda_t_norm**3)
                                )
                            J_newton[row_idx, lambda_t_start + 2 * c + j] = deriv
                    # ∂s/∂gamma
                    if lambda_t_norm > 1e-10:
                        J_newton[row_idx, gamma_start + c] = (
                            lambda_t[c, t_idx] / lambda_t_norm
                        )

            # 5. Derivative of friction complementarity w.r.t normal forces, tangential forces, and gamma
            for c in range(n_contacts):
                row_idx = gamma_start + c

                # ∂fb_t/∂gamma: Direct derivative
                dfb_dgamma = self.fb_derivative_a(
                    gamma[c], friction_bound[c] - lambda_t_norms[c]
                )
                J_newton[row_idx, gamma_start + c] = dfb_dgamma

                # ∂fb_t/∂lambda_n: Through friction bound
                dfb_dbound = self.fb_derivative_b(
                    gamma[c], friction_bound[c] - lambda_t_norms[c]
                )
                mu = model.dynamic_friction[contact_body[c]]
                J_newton[row_idx, lambda_n_start + c] = dfb_dbound * mu

                # ∂fb_t/∂lambda_t: Through norm of tangential forces
                if lambda_t_norms[c] > 1e-10:
                    for t_idx in range(2):
                        deriv = -dfb_dbound * lambda_t[c, t_idx] / lambda_t_norms[c]
                        J_newton[row_idx, lambda_t_start + 2 * c + t_idx] = deriv

            # Solve the Newton system
            debug.print("Solving Newton system")
            try:
                # Determine search direction: J_newton * dx = -residual
                dx = torch.linalg.solve(J_newton, -residual)
                debug.print("Direct solve successful")
            except:
                # Fallback to least-squares if direct solve fails
                debug.print("Direct solve failed, using least-squares")
                X = torch.linalg.lstsq(J_newton, -residual)
                dx = X.solution

            # Extract updates for different variables
            dv = dx[v_start:v_end]
            dlambda_n = dx[lambda_n_start:lambda_n_end]
            dlambda_t = dx[lambda_t_start:lambda_t_end].reshape(n_contacts, 2)
            dgamma = dx[gamma_start:gamma_end]

            debug.print("Update vector norms:")
            debug.indent()
            debug.print(f"dv: {torch.norm(dv):.6f}")
            debug.print(f"dlambda_n: {torch.norm(dlambda_n)}")
            debug.print(f"dlambda_t: {torch.norm(dlambda_t):.6f}")
            debug.print(f"dgamma: {torch.norm(dgamma):.6f}")
            debug.undent()

            # Apply updates with line search
            alpha = 1.0  # Line search parameter
            debug.print(f"Applying updates with alpha = {alpha}")

            # Update velocities
            v = v + alpha * dv

            # Update normal forces with non-negativity constraint
            lambda_n = torch.maximum(
                lambda_n + alpha * dlambda_n, torch.zeros_like(lambda_n)
            )

            # Update tangential forces
            lambda_t = lambda_t + alpha * dlambda_t

            # Update slack variables with non-negativity constraint
            gamma = torch.maximum(gamma + alpha * dgamma, torch.zeros_like(gamma))

            # Project tangential forces to friction cone if needed
            for c in range(n_contacts):
                lambda_t_norm = torch.norm(lambda_t[c])
                max_t = model.dynamic_friction[contact_body[c]] * lambda_n[c]
                if lambda_t_norm > max_t:
                    lambda_t[c] = lambda_t[c] * max_t / lambda_t_norm
                    debug.print(
                        f"Projected contact {c} tangential force to friction cone"
                    )
            debug.undent()

        # Summary of results
        debug.print("Contact solve complete")
        debug.print(f"Final normal forces: {lambda_n}")
        debug.print(f"Final tangential force norms: {torch.norm(lambda_t, dim=1)}")

        # Reshape velocities back to body format
        body_qd = v.reshape(n_bodies, 6)
        return body_qd

    def simulate(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        control: Control,
        dt: float,
    ):
        """Performs one simulation step using the non-smooth Newton method for contacts with friction.

        Args:
            model: The physics model
            state_in: The current state
            state_out: The output state to be computed
            control: The control inputs
            dt: Time step
        """
        debug = DebugLogger()
        debug.section(f"TIME: {state_in.time:.2f}")

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

        debug.print(f"Contact count: {model.contact_count}")

        # 1. Apply semi-implicit Euler integration for unconstrained step
        debug.section("UNCONSTRAINED INTEGRATION")
        for b in range(model.body_count):
            # Extract components
            x = body_q[b, :3].clone()  # Position
            q = body_q[b, 3:].clone()  # Rotation
            v = body_qd[b, 3:].clone()  # Linear velocity
            w = body_qd[b, :3].clone()  # Angular velocity
            t = body_f[b, :3].clone()  # Torque
            f = body_f[b, 3:].clone()  # Linear force
            c = torch.linalg.cross(w, model.body_inv_inertia[b] @ w)  # Coriolis force

            debug.print(f"- Body {b}:")
            debug.indent()
            debug.print("Initial Position:", x)
            debug.print("Initial Rotation:", quat_to_rotvec(q))
            debug.print("Initial Linear Velocity:", v)
            debug.print("Initial Angular Velocity:", w)
            debug.print("Applied Force:", f)
            debug.print("Applied Torque:", t)
            debug.print("Coriolis Force:", c)

            # Update velocities with forces
            v_new = v + (f * model.body_inv_mass[b] + model.gravity) * dt
            w_new = w + model.body_inv_inertia[b] @ (t - c) * dt

            debug.print("Updated Linear Velocity:", v_new)
            debug.print("Updated Angular Velocity:", w_new)
            debug.print("Linear Velocity Delta:", v_new - v)
            debug.print("Angular Velocity Delta:", w_new - w)
            debug.undent()

            # Store updated velocities
            body_qd[b, :3] = w_new
            body_qd[b, 3:] = v_new

        # 2. Solve contact and friction constraints using Newton's method
        debug.section("CONSTRAINT SOLVE")
        if model.contact_count > 0:
            # Create a temporary state with updated velocities
            temp_state = State()
            temp_state.body_q = body_q
            temp_state.body_qd = body_qd

            # Solve constraints
            body_qd = self.solve_contact_friction(model, temp_state, dt, debug)

            # Print velocity changes due to constraints
            debug.subsection("VELOCITY CHANGES DUE TO CONSTRAINTS")
            for b in range(model.body_count):
                debug.print(f"- Body {b}:")
                debug.indent()
                old_v = temp_state.body_qd[b, 3:]
                old_w = temp_state.body_qd[b, :3]
                new_v = body_qd[b, 3:]
                new_w = body_qd[b, :3]
                debug.print("Linear Velocity Change:", new_v - old_v)
                debug.print("Angular Velocity Change:", new_w - old_w)
                debug.undent()
        else:
            debug.print("No contacts, skipping constraint solve")

        # 3. Update positions using corrected velocities
        debug.section("POSITION UPDATE")
        for b in range(model.body_count):
            x = body_q[b, :3]
            q = body_q[b, 3:]
            v = body_qd[b, 3:]
            w = body_qd[b, :3]

            debug.print(f"- Body {b}:")
            debug.indent()

            # Update position
            new_x = x + v * dt
            debug.print("Position:", x)
            debug.print("Position Delta:", v * dt)
            debug.print("New Position:", new_x)

            # Update orientation using quaternion integration
            w_q = torch.cat([torch.tensor([0.0], device=w.device), w])
            delta_q = 0.5 * quat_mul(w_q, q) * dt
            new_q = q + delta_q
            new_q = normalize_quat(new_q)

            debug.print("Rotation:", quat_to_rotvec(q))
            debug.print("Rotation Delta:", quat_to_rotvec(delta_q))
            debug.print("New Rotation:", quat_to_rotvec(new_q))
            debug.undent()

            body_q[b, :3] = new_x
            body_q[b, 3:] = new_q

        # Save the final state
        state_out.body_q = body_q
        state_out.body_qd = body_qd
        state_out.time = state_in.time + dt
