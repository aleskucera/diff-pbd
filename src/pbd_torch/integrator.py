from typing import Tuple

import torch
from pbd_torch.transform import normalize_quat
from pbd_torch.transform import normalize_quat_batch
from pbd_torch.transform import quat_mul
from pbd_torch.transform import quat_mul_batch
from pbd_torch.transform import rotate_vector
from pbd_torch.transform import rotate_vector_inverse
from pbd_torch.transform import rotate_vectors_batch
from pbd_torch.transform import rotate_vectors_inverse_batch


class SemiImplicitEulerIntegrator:
    def __init__(self, use_local_omega: bool = False, device=None):
        """
        Initialize the integrator.

        Args:
            use_local_omega (bool): If True, angular velocity is in local (body) frame.
                                  If False, angular velocity is in global (world) frame.
            device: The torch device to use.
        """
        self.use_local_omega = use_local_omega
        self.device = device

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
        """
        Integrate a single body.

        Args:
            body_q: Position and rotation quaternion [x,y,z, qw,qx,qy,qz]
            body_qd: Linear and angular velocity [wx,wy,wz, vx,vy,vz]
            body_f: Force and torque [tx,ty,tz, fx,fy,fz]
            body_inv_mass: Inverse mass
            body_inv_inertia: Inverse inertia tensor (3x3)
            gravity: Gravity vector [gx,gy,gz]
            dt: Time step

        Returns:
            Tuple of updated position & rotation and velocities
        """
        if self.use_local_omega:
            return self._integrate_body_local_omega(
                body_q, body_qd, body_f, body_inv_mass, body_inv_inertia, gravity, dt
            )
        else:
            return self._integrate_body_global_omega(
                body_q, body_qd, body_f, body_inv_mass, body_inv_inertia, gravity, dt
            )

    def integrate(
        self,
        body_q: torch.Tensor,
        body_qd: torch.Tensor,
        body_f: torch.Tensor,
        body_inv_mass: torch.Tensor,
        body_inv_inertia: torch.Tensor,
        gravity: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate multiple bodies simultaneously.

        Args:
            body_q: Batch of positions and rotation quaternions [N, 7]
            body_qd: Batch of linear and angular velocities [N, 6]
            body_f: Batch of forces and torques [N, 6]
            body_inv_mass: Batch of inverse masses [N]
            body_inv_inertia: Batch of inverse inertia tensors [N, 3, 3]
            gravity: Gravity vector [3]
            dt: Time step

        Returns:
            Tuple of updated positions & rotations and velocities for all bodies
        """
        if self.use_local_omega:
            return self._integrate_bodies_vectorized_local_omega(
                body_q, body_qd, body_f, body_inv_mass, body_inv_inertia, gravity, dt
            )
        else:
            return self._integrate_bodies_vectorized_global_omega(
                body_q, body_qd, body_f, body_inv_mass, body_inv_inertia, gravity, dt
            )

    def _integrate_body_global_omega(
        self,
        body_q: torch.Tensor,
        body_qd: torch.Tensor,
        body_f: torch.Tensor,
        body_inv_mass: torch.Tensor,
        body_inv_inertia: torch.Tensor,
        gravity: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrates a single body with angular velocity in world frame."""
        x0 = body_q[:3]  # Position
        q0 = body_q[3:]  # Rotation (body to world)
        v0 = body_qd[3:]  # Linear velocity (world frame)
        w0 = body_qd[:3]  # Angular velocity (world frame)
        t0 = body_f[:3]  # Torque (body frame)
        f0 = body_f[3:]  # Linear force (world frame)

        # Transform world angular velocity to body frame for dynamics calculation
        w0_body = rotate_vector_inverse(w0, q0)

        # Compute Coriolis force in body frame
        c = torch.linalg.cross(w0_body, body_inv_inertia @ w0_body)

        # Compute angular acceleration in body frame
        alpha_body = torch.matmul(body_inv_inertia, t0 - c)

        # Transform angular acceleration back to world frame
        alpha_world = rotate_vector(alpha_body, q0)

        # Integrate linear velocity and position
        v1 = v0 + (f0 * body_inv_mass + gravity) * dt
        x1 = x0 + v1 * dt

        # Integrate angular velocity in world frame
        w1 = w0 + alpha_world * dt

        # Use world frame angular velocity directly for quaternion integration
        q1 = (
            q0
            + 0.5
            * quat_mul(torch.cat([torch.tensor([0.0], device=w1.device), w1]), q0)
            * dt
        )
        q1 = normalize_quat(q1)

        new_body_q = torch.cat([x1, q1])
        new_body_qd = torch.cat([w1, v1])

        return new_body_q, new_body_qd

    def _integrate_bodies_vectorized_global_omega(
        self,
        body_q: torch.Tensor,
        body_qd: torch.Tensor,
        body_f: torch.Tensor,
        body_inv_mass: torch.Tensor,
        body_inv_inertia: torch.Tensor,
        gravity: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrates all bodies with angular velocities in world frame."""

        # Extract components
        x0 = body_q[:, :3]  # Positions [N, 3]
        q0 = body_q[:, 3:]  # Rotations [N, 4]
        v0 = body_qd[:, 3:]  # Linear velocities [N, 3]
        w0 = body_qd[:, :3]  # Angular velocities (world frame) [N, 3]
        t0 = body_f[:, :3]  # Torques (body frame) [N, 3]
        f0 = body_f[:, 3:]  # Linear forces [N, 3]

        # Create inverse quaternions (for world to body rotation)
        q0_inv = torch.cat([q0[:, 0:1], -q0[:, 1:]], dim=1)

        # Transform world angular velocities to body frame
        w0_body = rotate_vectors_batch(w0, q0_inv)

        # Compute Coriolis terms in body frame
        I_w = torch.bmm(body_inv_inertia, w0_body.unsqueeze(2)).squeeze(2)
        c = torch.cross(w0_body, I_w, dim=1)

        # Compute angular acceleration in body frame
        alpha_body = torch.bmm(body_inv_inertia, (t0 - c).unsqueeze(2)).squeeze(2)

        # Transform angular acceleration back to world frame
        alpha_world = rotate_vectors_batch(alpha_body, q0)

        # Integrate linear motion
        v1 = v0 + (f0 * body_inv_mass.unsqueeze(1) + gravity.unsqueeze(0)) * dt
        x1 = x0 + v1 * dt

        # Integrate angular velocity in world frame
        w1 = w0 + alpha_world * dt

        # Quaternion integration using world frame angular velocities directly
        zeros = torch.zeros(w1.shape[0], 1, device=w1.device)
        w1_quat = torch.cat([zeros, w1], dim=1)  # [N, 4]

        q_dot = 0.5 * quat_mul_batch(w1_quat, q0)
        q1 = q0 + q_dot * dt
        q1 = normalize_quat_batch(q1)

        # Assemble final states
        new_body_q = torch.cat([x1, q1], dim=1)
        new_body_qd = torch.cat([w1, v1], dim=1)

        return new_body_q, new_body_qd

    def _integrate_body_local_omega(
        self,
        body_q: torch.Tensor,
        body_qd: torch.Tensor,
        body_f: torch.Tensor,
        body_inv_mass: torch.Tensor,
        body_inv_inertia: torch.Tensor,
        gravity: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrates a single body with angular velocity in body frame."""
        x0 = body_q[:3]  # Position
        q0 = body_q[3:]  # Rotation (body to world)
        v0 = body_qd[3:]  # Linear velocity (world frame)
        w0 = body_qd[:3]  # Angular velocity (body frame)
        t0 = body_f[:3]  # Torque (body frame)
        f0 = body_f[3:]  # Linear force (world frame)

        # Coriolis force (in body frame)
        c = torch.linalg.cross(w0, body_inv_inertia @ w0)

        # Integrate the velocity and position
        v1 = v0 + (f0 * body_inv_mass + gravity) * dt
        x1 = x0 + v1 * dt

        # Integrate the angular velocity and orientation
        w1 = w0 + torch.matmul(body_inv_inertia, t0 - c) * dt

        # Convert local angular velocity to world frame for quaternion integration
        w1_w = rotate_vector(w1, q0)

        q1 = (
            q0
            + 0.5
            * quat_mul(torch.cat([torch.tensor([0.0], device=w1_w.device), w1_w]), q0)
            * dt
        )
        q1 = normalize_quat(q1)

        new_body_q = torch.cat([x1, q1])
        new_body_qd = torch.cat([w1, v1])

        return new_body_q, new_body_qd

    def _integrate_bodies_vectorized_local_omega(
        self,
        body_q: torch.Tensor,
        body_qd: torch.Tensor,
        body_f: torch.Tensor,
        body_inv_mass: torch.Tensor,
        body_inv_inertia: torch.Tensor,
        gravity: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrates all bodies with angular velocities in local/body frame."""
        # Extract components for all bodies at once
        x0 = body_q[:, :3]  # Positions [N, 3, 1]
        q0 = body_q[:, 3:]  # Rotations [N, 4, 1]
        v0 = body_qd[:, 3:]  # Linear velocities [N, 3, 1]
        w0 = body_qd[:, :3]  # Angular velocities (local frame) [N, 3, 1]
        t0 = body_f[:, :3]  # Torques [N, 3, 1]
        f0 = body_f[:, 3:]  # Linear forces [N, 3, 1]

        # Integrate linear velocity and position for all bodies
        v1 = v0 + (f0 * body_inv_mass + gravity[3:]) * dt # [N, 3, 1]
        x1 = x0 + v1 * dt # [N, 3, 1]

        # Coriolis force and angular velocity integration in local frame
        I_w = torch.bmm(body_inv_inertia, w0)  # [N, 3, 1]
        c = torch.cross(w0, I_w, dim=1)  # [N, 3, 1]

        # Angular velocity integration (still in local frame)
        w1 = w0 + torch.bmm(body_inv_inertia, t0 - c) * dt # [N, 3, 1]

        # Convert local angular velocities to world frame for quaternion integration
        w1_w = rotate_vectors_batch(w1, q0)  # [N, 3, 1]

        # Create quaternions from angular velocities
        zeros = torch.zeros((w1_w.shape[0], 1, 1), device=w1_w.device)
        w1_w_quat = torch.cat([zeros, w1_w], dim=1)  # [N, 4, 1]

        # Batch quaternion multiplication
        q_dot = 0.5 * quat_mul_batch(w1_w_quat, q0) # [N, 4, 1]

        q1 = q0 + q_dot * dt # [N, 4, 1]

        # Normalize all quaternions at once
        q1_norm = normalize_quat_batch(q1) # [N, 4, 1]

        # Assemble final states
        new_body_q = torch.cat([x1, q1_norm], dim=1) # [N, 7, 1]
        new_body_qd = torch.cat([w1, v1], dim=1) # [N, 6, 1]

        return new_body_q, new_body_qd
