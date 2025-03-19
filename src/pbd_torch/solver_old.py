import os
from typing import Tuple

import torch
from pbd_torch.model import Control
from pbd_torch.model import Model
from pbd_torch.model import State
from pbd_torch.printer import DebugLogger
from pbd_torch.transform import *

os.environ["DEBUG"] = "true"
# os.environ["DEBUG"] = "false"


class VelocityLevelIntegrator:
    def __init__(self, iterations: int = 10):
        """Initialize the velocity-level integrator.

        Args:
            iterations (int): Number of constraint iterations per simulation step
        """
        self.iterations = iterations
        # Default Baumgarte stabilization parameter (0-1)
        self.baumgarte_factor = 0.2

    def integrate_body(
        self,
        body_q: torch.Tensor,
        body_qd: torch.Tensor,
        body_f: torch.Tensor,
        body_inv_mass: torch.Tensor,
        body_inv_inertia: torch.Tensor,
        gravity: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrates the body's position and orientation using semi-implicit Euler.

        Args:
            Same as in your original integrator

        Returns:
            torch.Tensor: Updated body velocities
        """
        # Semi-implicit Euler integration
        # First update velocity with forces
        x0 = body_q[:3].clone()  # Position
        q0 = body_q[3:].clone()  # Rotation
        v0 = body_qd[3:].clone()  # Linear velocity
        w0 = body_qd[:3].clone()  # Angular velocity
        t0 = body_f[:3].clone()  # Torque
        f0 = body_f[3:].clone()  # Linear force
        c = torch.linalg.cross(w0, body_inv_inertia @ w0)  # Coriolis force

        # Update velocities with forces (this will be further modified by constraints)
        v1 = v0 + (f0 * body_inv_mass + gravity) * dt
        w1 = w0 + torch.matmul(body_inv_inertia, t0 - c) * dt

        # Return velocities but don't update positions yet
        # Positions will be updated after constraint solving
        new_body_qd = torch.cat([w1, v1])
        return new_body_qd

    def solve_contact_constraints(
        self,
        body_q: torch.Tensor,
        body_qd: torch.Tensor,
        contact_count: int,
        contact_body: torch.Tensor,
        contact_point: torch.Tensor,
        contact_normal: torch.Tensor,
        contact_point_ground: torch.Tensor,
        body_inv_mass: torch.Tensor,
        body_inv_inertia: torch.Tensor,
        restitution: torch.Tensor,
        dynamic_friction: torch.Tensor,
        dt: float,
        debug: DebugLogger,
    ) -> torch.Tensor:
        """Solve contact constraints at the velocity level.

        Args:
            body_q: Body positions and orientations
            body_qd: Body linear and angular velocities
            contact_count: Number of contacts
            contact_body: Body indices for each contact
            contact_point: Contact points in body coordinates
            contact_normal: Contact normal directions
            contact_point_ground: Contact points on ground
            body_inv_mass: Inverse masses
            body_inv_inertia: Inverse inertia tensors
            restitution: Restitution coefficients
            dynamic_friction: Dynamic friction coefficients
            dt: Time step
            debug: Debug printer

        Returns:
            torch.Tensor: Updated body velocities
        """
        if contact_count == 0:
            return body_qd

        debug.print("Contact Count:", contact_count)

        # Copy velocities to modify
        updated_body_qd = body_qd.clone()

        # Track impulses per body for debugging
        normal_impulses = torch.zeros((body_q.shape[0]), device=body_q.device)
        friction_impulses = torch.zeros((body_q.shape[0]), device=body_q.device)

        # Loop through iterations to solve constraints
        for iter_idx in range(self.iterations):
            debug.subsection(f"CONSTRAINT ITERATION {iter_idx + 1}")
            debug.print("- NORMAL CONTACT:")
            debug.indent()

            for c in range(contact_count):
                b = contact_body[c].item()  # Body index for this contact
                r = contact_point[c]  # Vector from body center to contact point
                n = -contact_normal[c]  # Contact normal

                # Skip if not in contact
                r_world = body_to_world(r, body_q[b])
                r_ground = contact_point_ground[c]
                if r_world[2] >= 0:
                    continue

                # 1. Normal contact constraint
                # --------------------------

                # Compute relative velocity at contact point
                w = updated_body_qd[b, :3]  # Angular velocity
                v = updated_body_qd[b, 3:]  # Linear velocity
                v_rel = v + torch.linalg.cross(w, r)

                # Normal component of relative velocity
                v_normal = torch.dot(v_rel, n)

                # For penetrating contacts, compute bias term for position correction
                penetration_depth = (r_ground - r_world).dot(n)
                bias = self.baumgarte_factor * penetration_depth / dt

                # Add restitution effect (if moving toward the surface)
                restitution_term = 0.0
                if v_normal < 0:
                    restitution_term = restitution[b] * (v_normal)
                    bias += restitution_term

                # Compute J*v + b
                constraint_value = v_normal + bias

                debug.print(f"Body {b}, Contact {c}:")
                debug.indent()
                debug.print(f"Penetration: {penetration_depth:.6f}")
                debug.print(f"Normal Velocity: {v_normal:.6f}")
                debug.print(
                    f"Bias: {bias:.6f} (Baumgarte: {bias - restitution_term:.6f}, Restitution: {restitution_term:.6f})"
                )
                debug.print(f"Constraint Value: {constraint_value:.6f}")

                if constraint_value < 0:
                    # Compute Jacobian for normal constraint: J = [-n^T, -(r×n)^T]
                    J_linear = n
                    J_angular = torch.linalg.cross(r, n)

                    # Compute K = J*M^-1*J^T
                    K = body_inv_mass[b] + torch.dot(
                        torch.linalg.cross(
                            body_inv_inertia[b] @ torch.linalg.cross(r, n), r
                        ),
                        n,
                    )

                    # Compute normal impulse: λ = -constraint/(J*M^-1*J^T)
                    lambda_n = -constraint_value / K if abs(K) > 1e-10 else 0.0
                    normal_impulses[b] += lambda_n

                    # Correct velocities: v += M^-1*J^T*λ
                    delta_v = J_linear * lambda_n * body_inv_mass[b]
                    delta_w = body_inv_inertia[b] @ (J_angular * lambda_n)

                    debug.print(f"K: {K:.6f}")
                    debug.print(f"Normal Impulse: {lambda_n:.6f}")
                    debug.print(f"Delta Linear Velocity: {delta_v}")
                    debug.print(f"Delta Angular Velocity: {delta_w}")

                    updated_body_qd[b, 3:] += delta_v
                    updated_body_qd[b, :3] += delta_w

                    # 2. Friction constraints
                    # ----------------------

                    # Only apply friction if there was a normal impulse
                    if lambda_n > 0 and dynamic_friction[b] > 0:
                        debug.print("- FRICTION:")
                        debug.indent()

                        # Recompute relative velocity after normal correction
                        v = updated_body_qd[b, 3:]
                        w = updated_body_qd[b, :3]
                        v_rel = v + torch.linalg.cross(w, r)

                        # Tangential velocity
                        v_normal_component = torch.dot(v_rel, n) * n
                        v_tangent = v_rel - v_normal_component
                        v_tangent_mag = torch.norm(v_tangent)

                        debug.print(
                            f"Tangential Velocity Magnitude: {v_tangent_mag:.6f}"
                        )

                        if v_tangent_mag > 1e-6:
                            # Tangent direction
                            t = v_tangent / v_tangent_mag

                            # Compute friction Jacobian: J = [-t^T, -(r×t)^T]
                            J_linear_friction = -t
                            J_angular_friction = -torch.linalg.cross(r, t)

                            # Compute K for friction
                            K_friction = body_inv_mass[b] + torch.dot(
                                torch.linalg.cross(
                                    body_inv_inertia[b] @ torch.linalg.cross(r, t), r
                                ),
                                t,
                            )

                            # Compute max friction impulse allowed by Coulomb's law
                            max_friction_impulse = lambda_n * dynamic_friction[b]

                            # Compute friction impulse
                            lambda_friction = (
                                v_tangent_mag / K_friction
                                if abs(K_friction) > 1e-10
                                else 0.0
                            )
                            lambda_friction = min(lambda_friction, max_friction_impulse)
                            friction_impulses[b] += lambda_friction

                            # Apply friction impulse
                            delta_v_friction = (
                                J_linear_friction * lambda_friction * body_inv_mass[b]
                            )
                            delta_w_friction = body_inv_inertia[b] @ (
                                J_angular_friction * lambda_friction
                            )

                            debug.print(
                                f"Friction Coefficient: {dynamic_friction[b]:.4f}"
                            )
                            debug.print(
                                f"Max Allowed Impulse: {max_friction_impulse:.6f}"
                            )
                            debug.print(
                                f"Actual Friction Impulse: {lambda_friction:.6f}"
                            )
                            debug.print(
                                f"Delta Friction Linear Velocity: {delta_v_friction}"
                            )
                            debug.print(
                                f"Delta Friction Angular Velocity: {delta_w_friction}"
                            )

                            updated_body_qd[b, 3:] += delta_v_friction
                            updated_body_qd[b, :3] += delta_w_friction
                        else:
                            debug.print(
                                "Tangential velocity too small, skipping friction"
                            )

                        debug.undent()
                else:
                    debug.print("No constraint violation, skipping")

                debug.undent()  # end of contact details

            debug.undent()  # end of normal contact section

        # Print the total impulses applied to each body
        debug.print("\n- CONTACT SUMMARY:")
        debug.indent()
        for b in range(body_q.shape[0]):
            if normal_impulses[b] > 0 or friction_impulses[b] > 0:
                debug.print(f"Body {b}:")
                debug.indent()
                debug.print(f"Total Normal Impulse: {normal_impulses[b]:.6f}")
                debug.print(f"Total Friction Impulse: {friction_impulses[b]:.6f}")
                debug.undent()
        debug.undent()

        return updated_body_qd

    def update_positions(
        self,
        body_q: torch.Tensor,
        body_qd: torch.Tensor,
        dt: float,
        debug: DebugLogger,
    ) -> torch.Tensor:
        """Update positions and orientations using the corrected velocities.

        Args:
            body_q: Current positions and orientations
            body_qd: Corrected velocities
            dt: Time step
            debug: Debug printer

        Returns:
            torch.Tensor: Updated positions and orientations
        """
        debug.print("- POSITION UPDATE:")
        debug.indent()

        new_body_q = body_q.clone()

        for b in range(body_q.shape[0]):
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

            debug.print(f"Body {b}:")
            debug.indent()
            debug.print(f"Old Position: {x}")
            debug.print(f"Old Rotation: {quat_to_rotvec(q)}")
            debug.print(f"New Position: {new_x}")
            debug.print(f"New Rotation: {quat_to_rotvec(new_q)}")
            debug.print(f"Delta Position: {new_x - x}")
            debug.print(f"Delta Rotation: {quat_to_rotvec(new_q) - quat_to_rotvec(q)}")
            debug.undent()

            new_body_q[b, :3] = new_x
            new_body_q[b, 3:] = new_q

        debug.undent()
        return new_body_q

    def simulate(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        control: Control,
        dt: float,
    ):
        """Performs one simulation step using velocity-level constraints.

        Args:
            model: The physics model
            state_in: The current state
            state_out: The output state to be computed
            control: The control inputs
            dt: Time step
        """
        debug = DebugLogger()
        debug.section(f"TIME: {state_in.time:.4f}")

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

        # ======================================== START: INTEGRATION ========================================
        debug.section("UNCONSTRAINED INTEGRATION")

        # 1. Apply external forces and compute unconstrained velocities
        for b in range(model.body_count):
            old_velocity = body_qd[b].clone()
            body_qd[b] = self.integrate_body(
                body_q[b],
                body_qd[b],
                body_f[b],
                model.body_inv_mass[b],
                model.body_inv_inertia[b],
                model.gravity,
                dt,
            )

            debug.print(f"- Body {b}:")
            debug.indent()
            debug.print("Old Linear Velocity:", old_velocity[3:])
            debug.print("Old Angular Velocity:", old_velocity[:3])
            debug.print("New Linear Velocity:", body_qd[b][3:])
            debug.print("New Angular Velocity:", body_qd[b][:3])
            debug.print("Delta Linear Velocity:", body_qd[b][3:] - old_velocity[3:])
            debug.print("Delta Angular Velocity:", body_qd[b][:3] - old_velocity[:3])
            debug.undent()

        # ======================================== START: CONSTRAINT SOLVING ========================================
        debug.section("CONSTRAINT SOLVING")

        # 2. Solve velocity constraints
        if model.contact_count > 0:
            body_qd = self.solve_contact_constraints(
                body_q,
                body_qd,
                model.contact_count,
                model.contact_body,
                model.contact_point,
                model.contact_normal,
                model.contact_point_ground,
                model.body_inv_mass,
                model.body_inv_inertia,
                model.restitution,
                model.dynamic_friction,
                dt,
                debug,
            )
        else:
            debug.print("No contacts detected, skipping constraint solving")

        # ======================================== START: POSITION UPDATE ========================================
        debug.section("POSITION UPDATE")

        # 3. Update positions using corrected velocities
        body_q = self.update_positions(body_q, body_qd, dt, debug)

        # Save the final state
        state_out.body_q = body_q
        state_out.body_qd = body_qd
        state_out.time = state_in.time + dt
