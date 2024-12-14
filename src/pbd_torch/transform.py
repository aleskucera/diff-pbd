import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    q = [w, x, y, z]
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1).type(q1.dtype)


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """
    Compute the conjugate of a quaternion.
    q = [w, x, y, z] -> q* = [w, -x, -y, -z]
    """
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1).type(q.dtype)


def rotate_vectors(points: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector v by quaternion q.
    v_rotated = q * v * q*
    """
    v_quat = torch.cat([torch.zeros_like(points[..., :1]), points], dim=-1)
    temp = quat_mul(q, v_quat)
    q_conj = quat_inv(q)
    rotated = quat_mul(temp, q_conj)
    return rotated[..., 1:].type(points.dtype)


def rotate_vectors_inverse(points: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector v by the inverse of quaternion q.
    v_rotated = q* * v * q
    """
    v_quat = torch.cat([torch.zeros_like(points[..., :1]), points], dim=-1)
    q_inv = quat_inv(q)
    temp = quat_mul(q_inv, v_quat)
    rotated = quat_mul(temp, q)
    return rotated[..., 1:].type(points.dtype)


def rotvec_to_quat(rotation_vector: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation vector to quaternion.
    rotation_vector = angle * axis, where axis is normalized.
    quaternion = [w, x, y, z]

    Args:
        rotation_vector: tensor of shape [..., 3]
    Returns:
        quaternion: tensor of shape [..., 4]
    """
    angle = torch.norm(rotation_vector, dim=-1, keepdim=True)

    # Handle zero rotation
    mask = angle.squeeze(-1) < 1e-8
    if mask.any():
        quat = torch.zeros((*rotation_vector.shape[:-1], 4),
                           device=rotation_vector.device)
        quat[..., 0] = 1.0  # w component = 1 for zero rotation
        if mask.all():
            return quat

    axis = rotation_vector / (angle + 1e-8
                              )  # add epsilon to avoid division by zero

    half_angle = angle / 2
    sin_half = torch.sin(half_angle)

    quat = torch.zeros((*rotation_vector.shape[:-1], 4),
                       device=rotation_vector.device)
    quat[..., 0] = torch.cos(half_angle).squeeze(-1)  # w
    quat[..., 1:] = axis * sin_half

    # Handle zero rotation case
    if mask.any():
        quat[mask, 0] = 1.0
        quat[mask, 1:] = 0.0

    return quat.type(rotation_vector.dtype)


def quat_to_rotvec(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation vector.
    quaternion = [w, x, y, z]
    rotation_vector = angle * axis

    Args:
        quaternion: tensor of shape [..., 4]
    Returns:
        rotation_vector: tensor of shape [..., 3]
    """
    # Normalize quaternion
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)

    # Extract w component and vector part
    w = quaternion[..., 0]
    xyz = quaternion[..., 1:]

    # Calculate angle
    angle = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))

    # Handle small angles to avoid numerical instability
    small_angle_mask = angle < 1e-8

    # Calculate axis
    sin_half_angle = torch.sqrt(1 - w * w + 1e-8)  # add epsilon to avoid nan
    axis = xyz / (sin_half_angle.unsqueeze(-1) + 1e-8)

    # Calculate rotation vector
    rotation_vector = axis * angle.unsqueeze(-1)

    # Handle small angles
    if small_angle_mask.any():
        rotation_vector[small_angle_mask] = xyz[small_angle_mask] * 2

    return rotation_vector.type(quaternion.dtype)


def quat_to_rotmat(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.
    quaternion = [w, x, y, z]

    Args:
        quaternion: tensor of shape [..., 4]
    Returns:
        rotation_matrix: tensor of shape [..., 3, 3]
    """
    # Normalize quaternion
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)

    # Extract w component and vector part
    w = quaternion[..., 0]
    x = quaternion[..., 1]
    y = quaternion[..., 2]
    z = quaternion[..., 3]

    # Calculate rotation matrix
    rotation_matrix = torch.stack([
        1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w,
        2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w,
        2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2
    ],
                                  dim=-1).reshape(*quaternion.shape[:-1], 3, 3)

    return rotation_matrix.type(quaternion.dtype)


def rotmat_to_quat(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion.
    rotation_matrix = [r00, r01, r02, r10, r11, r12, r20, r21, r22]
    quaternion = [w, x, y, z]

    Args:
        rotation_matrix: tensor of shape [..., 3, 3]
    Returns:
        quaternion: tensor of shape [..., 4]
    """
    # Extract rotation matrix elements
    r00 = rotation_matrix[..., 0, 0]
    r01 = rotation_matrix[..., 0, 1]
    r02 = rotation_matrix[..., 0, 2]
    r10 = rotation_matrix[..., 1, 0]
    r11 = rotation_matrix[..., 1, 1]
    r12 = rotation_matrix[..., 1, 2]
    r20 = rotation_matrix[..., 2, 0]
    r21 = rotation_matrix[..., 2, 1]
    r22 = rotation_matrix[..., 2, 2]

    # Calculate quaternion
    trace = r00 + r11 + r22
    w = torch.sqrt(1 + trace) / 2
    x = torch.sign(r21 - r12) * torch.sqrt(1 + r00 - r11 - r22) / 2
    y = torch.sign(r02 - r20) * torch.sqrt(1 - r00 + r11 - r22) / 2
    z = torch.sign(r10 - r01) * torch.sqrt(1 - r00 - r11 + r22) / 2

    quaternion = torch.stack([w, x, y, z], dim=-1)

    return quaternion.type(rotation_matrix.dtype)


def relative_rotation(q_a: torch.Tensor, q_b: torch.Tensor):
    """Compute the relative rotation between two poses. This rotation can
    take vectors in the local frame of q_a and transform them to the local
    frame of q_b.
    """
    q_rel = quat_mul(q_b, quat_inv(q_a))
    return q_rel

def relative_transform(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
    """Compute the relative transform between two poses. This transform can
    take vectors in the local frame of q_a and transform them to the local
    frame of q_b.
    """
    x = rotate_vectors_inverse(q_a[:3] - q_b[:3], q_b[3:])
    q = quat_mul(q_b[3:], quat_inv(q_a[3:]))
    return torch.cat([x, q])

def delta_q_fixed(q_a: torch.Tensor, q_b: torch.Tensor):
    dq = quat_mul(q_a, quat_inv(q_b))
    return dq


def normalize_quat(q: torch.Tensor):
    return q / torch.norm(q)


def transform(points: torch.Tensor, q: torch.Tensor):
    return rotate_vectors(points, q[3:]) + q[:3]


def body_to_world(r: torch.Tensor, body_q: torch.Tensor) -> torch.Tensor:
    body_x = body_q[:3]
    body_rot = body_q[3:]
    return rotate_vectors(r, body_rot) + body_x


def world_to_body(r: torch.Tensor, body_q: torch.Tensor) -> torch.Tensor:
    body_x = body_q[:3]
    body_rot = body_q[3:]
    return rotate_vectors_inverse(r - body_x, body_rot)


def body_to_robot(r: torch.Tensor, body_q: torch.Tensor,
                  robot_q: torch.Tensor) -> torch.Tensor:
    world_r = body_to_world(r, body_q)
    return world_to_body(world_r, robot_q)


def robot_to_body(r: torch.Tensor, robot_q: torch.Tensor,
                  body_q: torch.Tensor) -> torch.Tensor:
    world_r = body_to_world(r, body_q)
    return world_to_body(world_r, robot_q)

def body_to_robot_transform(body_q: torch.Tensor,
                            robot_q: torch.Tensor) -> torch.Tensor:
    return relative_transform(body_q, robot_q)

def robot_to_body_transform(robot_q: torch.Tensor,
                            body_q: torch.Tensor) -> torch.Tensor:
    return relative_transform(robot_q, body_q)
