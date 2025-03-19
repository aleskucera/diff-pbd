from typing import Tuple

import torch
from pbd_torch.transform import normalize_quat
from pbd_torch.transform import quat_mul
from pbd_torch.transform import rotate_vectors


def integrate_body(
    body_q: torch.Tensor,
    body_qd: torch.Tensor,
    body_f: torch.Tensor,
    body_inv_mass: torch.Tensor,
    body_inv_inertia: torch.Tensor,
    gravity: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integrates the body's position and orientation over a time step.

    Args:
        body_q (torch.Tensor): Current body state (position[3] and quaternion[4])
        body_qd (torch.Tensor): Current body velocities (angular[3] and linear[3])
        body_f (torch.Tensor): Forces and torques applied to the body
        body_inv_mass (torch.Tensor): Inverse mass of the body
        body_inv_inertia (torch.Tensor): Inverse inertia tensor
        gravity (torch.Tensor): Gravity vector
        dt (float): Time step

    Returns:
        tuple: Updated (body_q, body_qd)
    """

    x0 = body_q[:3].clone()  # Position
    q0 = body_q[3:].clone()  # Rotation
    v0 = body_qd[3:].clone()  # Linear velocity
    w0 = body_qd[:3].clone()  # Angular velocity
    t0 = body_f[:3].clone()  # Torque
    f0 = body_f[3:].clone()  # Linear force
    c = torch.linalg.cross(w0, body_inv_inertia @ w0)  # Coriolis force

    # Integrate the velocity and position
    v1 = v0 + (f0 * body_inv_mass + gravity) * dt
    x1 = x0 + v1 * dt

    # Integrate the angular velocity and orientation
    w1 = w0 + torch.matmul(body_inv_inertia, t0 - c) * dt
    w1_w = rotate_vectors(w1, q0)
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


def integrate_bodies_vectorized(
    body_q: torch.Tensor,
    body_qd: torch.Tensor,
    body_f: torch.Tensor,
    body_inv_mass: torch.Tensor,
    body_inv_inertia: torch.Tensor,
    gravity: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integrates all bodies' positions and orientations over a time step in fully vectorized form.

    Args:
        body_q (torch.Tensor): Current body states (position[3] and quaternion[4]) for all bodies [N, 7]
        body_qd (torch.Tensor): Current body velocities (angular[3] and linear[3]) for all bodies [N, 6]
        body_f (torch.Tensor): Forces and torques applied to all bodies [N, 6]
        body_inv_mass (torch.Tensor): Inverse mass of all bodies [N]
        body_inv_inertia (torch.Tensor): Inverse inertia tensor for all bodies [N, 3, 3]
        gravity (torch.Tensor): Gravity vector [3]
        dt (float): Time step

    Returns:
        tuple: Updated (body_q, body_qd) for all bodies
    """
    # Extract components for all bodies at once
    x0 = body_q[:, :3].clone()  # Positions [N, 3]
    q0 = body_q[:, 3:].clone()  # Rotations [N, 4]
    v0 = body_qd[:, 3:].clone()  # Linear velocities [N, 3]
    w0 = body_qd[:, :3].clone()  # Angular velocities [N, 3]
    t0 = body_f[:, :3].clone()  # Torques [N, 3]
    f0 = body_f[:, 3:].clone()  # Linear forces [N, 3]

    # Integrate linear velocity and position for all bodies
    v1 = v0 + (f0 * body_inv_mass.unsqueeze(1) + gravity.unsqueeze(0)) * dt
    x1 = x0 + v1 * dt

    # Coriolis force and angular velocity integration
    # For each body i: c_i = w_i × (I_inv_i @ w_i)
    # For each body i: w1_i = w0_i + I_inv_i @ (t0_i - c_i) * dt

    # Compute w_i × (I_inv_i @ w_i) for all bodies at once
    # We'll use batch matrix multiplication for I_inv @ w
    I_w = torch.bmm(body_inv_inertia, w0.unsqueeze(2)).squeeze(2)  # [N, 3]
    c = torch.cross(w0, I_w, dim=1)  # [N, 3]

    # Angular velocity integration
    w1 = (
        w0 + torch.bmm(body_inv_inertia, (t0 - c).unsqueeze(2)).squeeze(2) * dt
    )  # [N, 3]

    # Quaternion integration using the rotated angular velocities
    # Batch-rotate all angular velocities
    w1_w = batch_rotate_vectors(w1, q0)  # [N, 3]

    # Create quaternions from angular velocities
    zeros = torch.zeros(w1_w.shape[0], 1, device=w1_w.device)
    w1_w_quat = torch.cat([zeros, w1_w], dim=1)  # [N, 4]

    # Batch quaternion multiplication
    q_dot = 0.5 * quat_mul(w1_w_quat, q0)

    q1 = q0 + q_dot * dt

    # Normalize all quaternions at once
    q1_norm = normalize_quat(q1)

    # Assemble final states
    new_body_q = torch.cat([x1, q1_norm], dim=1)
    new_body_qd = torch.cat([w1, v1], dim=1)

    return new_body_q, new_body_qd


def batch_rotate_vectors(vectors: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """Rotates vectors by quaternions in a batched manner.

    Args:
        vectors (torch.Tensor): Vectors to rotate [N, 3]
        quats (torch.Tensor): Quaternions to rotate by [N, 4]

    Returns:
        torch.Tensor: Rotated vectors [N, 3]
    """
    # Extract quaternion components
    q_w = quats[:, 0]  # [N]
    q_v = quats[:, 1:]  # [N, 3]

    # Implement quaternion rotation formula: v' = q * v * q^(-1)
    # v' = 2.0 * dot(q_v, v) * q_v + (q_w*q_w - dot(q_v, q_v)) * v + 2.0 * q_w * cross(q_v, v)

    # Compute dot products for all vectors and quaternions
    q_v_dot_v = torch.sum(q_v * vectors, dim=1, keepdim=True)  # [N, 1]
    q_v_norm = torch.sum(q_v * q_v, dim=1, keepdim=True)  # [N, 1]

    # Compute cross products for all vectors and quaternions
    q_v_cross_v = torch.cross(q_v, vectors, dim=1)  # [N, 3]

    # Compute the formula
    term1 = 2.0 * q_v_dot_v * q_v
    term2 = (q_w.unsqueeze(1) ** 2 - q_v_norm) * vectors
    term3 = 2.0 * q_w.unsqueeze(1) * q_v_cross_v

    return term1 + term2 + term3
