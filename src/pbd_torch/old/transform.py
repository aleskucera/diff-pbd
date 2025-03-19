import torch


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

    q_out = torch.stack([w, x, y, z], dim=-1).type(q1.dtype)

    return q_out


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """
    Compute the conjugate of a quaternion.
    q = [w, x, y, z] -> q* = [w, -x, -y, -z]
    """
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1).type(q.dtype)


def rotate_vectors(vectors: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """
    Rotates vectors by quaternions using an efficient direct formula.
    Works for both single and batched inputs and preserves input types.

    Args:
        vectors (torch.Tensor): Vector(s) to rotate, shape [..., 3]
        quats (torch.Tensor): Quaternion(s) to rotate by, shape [..., 4] or [4]

    Returns:
        torch.Tensor: Rotated vector(s) with same shape and dtype as input vectors
    """
    # Save original dtype
    original_dtype = vectors.dtype

    # Convert inputs to the same dtype for computation
    compute_dtype = torch.promote_types(vectors.dtype, quats.dtype)
    vectors = vectors.to(compute_dtype)
    quats = quats.to(compute_dtype)

    # Handle case where quats is a single quaternion [4] but vectors has batch dimensions
    if vectors.ndim > 1 and quats.ndim == 1:
        # Expand quats to match vectors' batch dimensions
        expanded_shape = list(vectors.shape[:-1]) + [4]
        quats = quats.expand(expanded_shape)

    # Extract quaternion components
    q_w = quats[..., 0]  # [...]
    q_v = quats[..., 1:]  # [..., 3]

    # Compute dot products
    q_v_dot_v = torch.sum(q_v * vectors, dim=-1, keepdim=True)  # [..., 1]
    q_v_norm = torch.sum(q_v * q_v, dim=-1, keepdim=True)  # [..., 1]

    # Compute cross products
    q_v_cross_v = torch.linalg.cross(q_v, vectors, dim=-1)  # [..., 3]

    # Compute formula: v' = 2.0 * dot(q_v, v) * q_v + (q_w*q_w - dot(q_v, q_v)) * v + 2.0 * q_w * cross(q_v, v)
    term1 = 2.0 * q_v_dot_v * q_v
    term2 = (q_w.unsqueeze(-1) ** 2 - q_v_norm) * vectors
    term3 = 2.0 * q_w.unsqueeze(-1) * q_v_cross_v

    result = term1 + term2 + term3

    # Ensure result has the same dtype as original input
    return result.to(dtype=original_dtype)


def rotate_vectors_inverse(vectors: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """
    Rotates vectors by the inverse of quaternions using an efficient direct formula.
    Works for both single and batched inputs and preserves input types.

    Args:
        vectors (torch.Tensor): Vector(s) to rotate, shape [..., 3]
        quats (torch.Tensor): Quaternion(s) to rotate by (will be inverted), shape [..., 4] or [4]

    Returns:
        torch.Tensor: Rotated vector(s) with same shape and dtype as input vectors
    """
    # Save original dtype
    original_dtype = vectors.dtype

    # Convert inputs to the same dtype for computation
    compute_dtype = torch.promote_types(vectors.dtype, quats.dtype)
    vectors = vectors.to(compute_dtype)
    quats = quats.to(compute_dtype)

    # Handle case where quats is a single quaternion [4] but vectors has batch dimensions
    if vectors.ndim > 1 and quats.ndim == 1:
        # Expand quats to match vectors' batch dimensions
        expanded_shape = list(vectors.shape[:-1]) + [4]
        quats = quats.expand(expanded_shape)

    # Inverse of a unit quaternion is its conjugate: [q_w, -q_v]
    q_w = quats[..., 0]  # [...]
    q_v = -quats[..., 1:]  # [..., 3] (negated for conjugate)

    # Compute dot products
    q_v_dot_v = torch.sum(q_v * vectors, dim=-1, keepdim=True)  # [..., 1]
    q_v_norm = torch.sum(q_v * q_v, dim=-1, keepdim=True)  # [..., 1]

    # Compute cross products
    q_v_cross_v = torch.linalg.cross(q_v, vectors, dim=-1)  # [..., 3]

    # Compute formula: v' = 2.0 * dot(q_v, v) * q_v + (q_w*q_w - dot(q_v, q_v)) * v + 2.0 * q_w * cross(q_v, v)
    term1 = 2.0 * q_v_dot_v * q_v
    term2 = (q_w.unsqueeze(-1) ** 2 - q_v_norm) * vectors
    term3 = 2.0 * q_w.unsqueeze(-1) * q_v_cross_v

    result = term1 + term2 + term3

    # Ensure result has the same dtype as original input
    return result.to(dtype=original_dtype)


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
        quat = torch.zeros(
            (*rotation_vector.shape[:-1], 4), device=rotation_vector.device
        )
        quat[..., 0] = 1.0  # w component = 1 for zero rotation
        if mask.all():
            return quat

    axis = rotation_vector / (angle + 1e-8)  # add epsilon to avoid division by zero

    half_angle = angle / 2
    sin_half = torch.sin(half_angle)

    quat = torch.zeros((*rotation_vector.shape[:-1], 4), device=rotation_vector.device)
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
    rotation_matrix = torch.stack(
        [
            1 - 2 * y**2 - 2 * z**2,
            2 * x * y - 2 * z * w,
            2 * x * z + 2 * y * w,
            2 * x * y + 2 * z * w,
            1 - 2 * x**2 - 2 * z**2,
            2 * y * z - 2 * x * w,
            2 * x * z - 2 * y * w,
            2 * y * z + 2 * x * w,
            1 - 2 * x**2 - 2 * y**2,
        ],
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)

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


def relative_transform(body_q_a: torch.Tensor, body_q_b: torch.Tensor) -> torch.Tensor:
    """Compute the relative transform between two poses. This transform can
    take vectors in the local frame of q_a and transform them to the local
    frame of q_b.
    """
    x = rotate_vectors_inverse(body_q_a[:3] - body_q_b[:3], body_q_b[3:])
    q = quat_mul(body_q_b[3:], quat_inv(body_q_a[3:]))
    return torch.cat([x, q])


def delta_q_fixed(q_a: torch.Tensor, q_b: torch.Tensor):
    dq = quat_mul(q_a, quat_inv(q_b))
    return dq


def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    """
    Args:
        q (torch.Tensor): Quaternions to normalize [N, 4]

    Returns:
        torch.Tensor: Normalized quaternions [N, 4]
    """
    # Compute quaternion magnitudes
    if q.dim() == 1:
        norm = torch.norm(q, keepdim=True)
    else:
        norm = torch.norm(q, dim=1, keepdim=True)

    # Avoid division by zero
    eps = 1e-10
    norm = torch.maximum(norm, torch.ones_like(norm) * eps)

    return q / norm


def transform(points: torch.Tensor, q: torch.Tensor):
    return rotate_vectors(points, q[3:]) + q[:3]


def transform_batch(points: torch.Tensor, q: torch.Tensor):
    return rotate_vectors(points, q[:, 3:]) + q[:, :3]


def body_to_world(r: torch.Tensor, body_q: torch.Tensor) -> torch.Tensor:
    body_x = body_q[:3]
    body_rot = body_q[3:]
    return rotate_vectors(r, body_rot) + body_x


def world_to_body(r: torch.Tensor, body_q: torch.Tensor) -> torch.Tensor:
    body_x = body_q[:3]
    body_rot = body_q[3:]
    return rotate_vectors_inverse(r - body_x, body_rot)


def body_to_robot(
    r: torch.Tensor, body_q: torch.Tensor, robot_q: torch.Tensor
) -> torch.Tensor:
    world_r = body_to_world(r, body_q)
    return world_to_body(world_r, robot_q)


def robot_to_body(
    r: torch.Tensor, robot_q: torch.Tensor, body_q: torch.Tensor
) -> torch.Tensor:
    world_r = body_to_world(r, body_q)
    return world_to_body(world_r, robot_q)


def body_to_robot_transform(
    body_q: torch.Tensor, robot_q: torch.Tensor
) -> torch.Tensor:
    return relative_transform(body_q, robot_q)


def robot_to_body_transform(
    robot_q: torch.Tensor, body_q: torch.Tensor
) -> torch.Tensor:
    return relative_transform(robot_q, body_q)


def transform_multiply(body_q_a: torch.Tensor, body_q_b: torch.Tensor) -> torch.Tensor:
    # Extract positions and orientations
    x_a = body_q_a[:3]  # position of frame A
    q_a = body_q_a[3:]  # orientation of frame A
    x_b = body_q_b[:3]  # position of frame B
    q_b = body_q_b[3:]  # orientation of frame B

    # First combine the rotations
    q = quat_mul(q_a, q_b)

    # Then transform the position:
    # 1. Rotate frame B's position by frame A's rotation
    # 2. Add frame A's position
    x = x_a + rotate_vectors(x_b, q_a)

    return torch.cat([x, q])


def transform_batch(points: torch.Tensor, transforms: torch.Tensor) -> torch.Tensor:
    """Transform points batched."""
    batch_size = points.shape[0]
    positions = transforms[:, :3]  # [batch_size, 3]
    rotations = transforms[:, 3:]  # [batch_size, 4]

    # Rotate points
    rotated_points = rotate_vectors_batch(points, rotations)  # [batch_size, 3]

    # Translate points
    transformed_points = rotated_points + positions  # [batch_size, 3]

    return transformed_points


def rotate_vectors_batch(vectors: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions batched."""
    # Create pure quaternions from the vectors
    zeros = torch.zeros(vectors.shape[0], 1, device=vectors.device)
    vec_quats = torch.cat([zeros, vectors], dim=1)  # [batch_size, 4]

    # Compute q * v * q^-1
    quat_inv = quat_inv_batch(quats)
    temp = quat_mul_batch(quats, vec_quats)
    rotated_vec_quats = quat_mul_batch(temp, quat_inv)

    # Extract vector part
    return rotated_vec_quats[:, 1:]


def rotate_vectors_inverse_batch(
    vectors: torch.Tensor, quats: torch.Tensor
) -> torch.Tensor:
    """Rotate vectors by inverse of quaternions batched."""
    # Compute inverse quaternions
    quat_inv = quat_inv_batch(quats)

    # Rotate using the inverse quaternions
    return rotate_vectors_batch(vectors, quat_inv)


def quat_mul_batch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply quaternions batched."""
    a1, b1, c1, d1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    a2, b2, c2, d2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    return torch.stack(
        [
            a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
        ],
        dim=1,
    )


def quat_inv_batch(quats: torch.Tensor) -> torch.Tensor:
    """Compute inverse quaternions batched."""
    result = torch.empty_like(quats)
    result[:, 0] = quats[:, 0]
    result[:, 1:] = -quats[:, 1:]
    return result
