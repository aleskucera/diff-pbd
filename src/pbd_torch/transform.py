import torch


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    q = [w, x, y, z]

    Args:
        q1 (torch.Tensor): First quaternion, shape [4]
        q2 (torch.Tensor): Second quaternion, shape [4]

    Returns:
        torch.Tensor: Resulting quaternion, shape [4]
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    q_out = torch.stack([w, x, y, z])
    return q_out.type(q1.dtype)


def quat_mul_batch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply quaternions in batch.
    q = [w, x, y, z]

    Args:
        q1 (torch.Tensor): First quaternions, shape [:, 4, 1]
        q2 (torch.Tensor): Second quaternions, shape [:, 4, 1]

    Returns:
        torch.Tensor: Resulting quaternions, shape [:, 4, 1]
    """
    w1, x1, y1, z1 = q1.unbind(1)
    w2, x2, y2, z2 = q2.unbind(1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    q_out = torch.stack([w, x, y, z], dim=1)
    return q_out.type(q1.dtype)


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """
    Compute the conjugate of a quaternion (inverse if normalized).
    q = [w, x, y, z] -> q* = [w, -x, -y, -z]

    Args:
        q (torch.Tensor): Quaternion, shape [4]

    Returns:
        torch.Tensor: Inverse quaternion, shape [4]
    """
    return torch.tensor([q[0], -q[1], -q[2], -q[3]]).type(q.dtype)


def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    """
    Normalize a quaternion to unit length.

    Args:
        q (torch.Tensor): Quaternion, shape [4]

    Returns:
        torch.Tensor: Normalized quaternion, shape [4]
    """
    q_normalized = q / torch.norm(q)
    return q_normalized


def normalize_quat_batch(q: torch.Tensor) -> torch.Tensor:
    """
    Normalize quaternions to unit length in batch.

    Args:
        q (torch.Tensor): Quaternions, shape [..., 4]

    Returns:
        torch.Tensor: Normalized quaternions, shape [..., 4]
    """
    q_normalized = q / torch.norm(q, dim=1, keepdim=True)
    return q_normalized


def quat_inv_batch(q: torch.Tensor) -> torch.Tensor:
    """
    Compute the conjugate of quaternions in batch (inverse if normalized).
    q = [w, x, y, z] -> q* = [w, -x, -y, -z]

    Args:
        q (torch.Tensor): Quaternions, shape [..., 4]

    Returns:
        torch.Tensor: Inverse quaternions, shape [..., 4]
    """
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1).type(q.dtype)


def rotate_vector(vector: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """
    Rotates a single 3D vector by a quaternion.

    Args:
        vector (torch.Tensor): Vector to rotate, shape [3]
        quat (torch.Tensor): Quaternion to rotate by, shape [4]

    Returns:
        torch.Tensor: Rotated vector, shape [3]
    """
    # Convert to correct dtype
    dtype = torch.promote_types(vector.dtype, quat.dtype)
    vector = vector.to(dtype)
    quat = quat.to(dtype)

    q_w = quat[0]
    q_v = quat[1:]

    # Compute dot products
    q_v_dot_v = torch.sum(q_v * vector)
    q_v_norm = torch.sum(q_v * q_v)

    # Compute cross products
    q_v_cross_v = torch.linalg.cross(q_v, vector)

    # Compute formula: v' = 2.0 * dot(q_v, v) * q_v + (q_w*q_w - dot(q_v, q_v)) * v + 2.0 * q_w * cross(q_v, v)
    term1 = 2.0 * q_v_dot_v * q_v
    term2 = (q_w * q_w - q_v_norm) * vector
    term3 = 2.0 * q_w * q_v_cross_v

    result = term1 + term2 + term3
    return result.to(vector.dtype)


def rotate_vectors_batch(vectors: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """
    Rotates vectors by quaternions in batch.

    Args:
        vectors (torch.Tensor): Vectors to rotate, shape [..., 3, 1]
        quats (torch.Tensor): Quaternions to rotate by, shape [..., 4, 1]

    Returns:
        torch.Tensor: Rotated vectors, shape [..., 3, 1]
    """
    # Convert to correct dtype
    dtype = torch.promote_types(vectors.dtype, quats.dtype)
    vectors = vectors.to(dtype)
    quats = quats.to(dtype)

    # Extract quaternion components
    q_w = quats[..., 0, :].unsqueeze(-2)
    q_v = quats[..., 1:, :]

    # Compute dot products
    q_v_dot_v = torch.sum(q_v * vectors, dim=-2, keepdim=True) # [..., 1, 1]
    q_v_norm = torch.sum(q_v * q_v, dim=-2, keepdim=True) # [..., 1, 1]

    # Compute cross products
    q_v_cross_v = torch.linalg.cross(q_v, vectors, dim=-2) # [..., 3, 1]

    # Compute formula
    term1 = 2.0 * q_v_dot_v * q_v # [..., 3, 1]
    term2 = (q_w ** 2 - q_v_norm) * vectors # [..., 3, 1]
    term3 = 2.0 * q_w * q_v_cross_v # [..., 3, 1]

    result = term1 + term2 + term3
    return result.to(vectors.dtype)


def rotate_vector_inverse(vector: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """
    Rotates a single 3D vector by the inverse of a quaternion.

    Args:
        vector (torch.Tensor): Vector to rotate, shape [3]
        quat (torch.Tensor): Quaternion to rotate by (will be inverted), shape [4]

    Returns:
        torch.Tensor: Rotated vector, shape [3]
    """
    return rotate_vector(vector, quat_inv(quat))


def rotate_vectors_inverse_batch(
    vectors: torch.Tensor, quats: torch.Tensor
) -> torch.Tensor:
    """
    Rotates vectors by the inverse of quaternions in batch.

    Args:
        vectors (torch.Tensor): Vectors to rotate, shape [..., 3]
        quats (torch.Tensor): Quaternions to rotate by (will be inverted), shape [..., 4]

    Returns:
        torch.Tensor: Rotated vectors, shape [..., 3]
    """
    return rotate_vectors_batch(vectors, quat_inv_batch(quats))


def transform_point(point: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """
    Apply a rigid transform to a point.

    Args:
        point (torch.Tensor): Point to transform, shape [3]
        transform (torch.Tensor): Transform [x, y, z, qw, qx, qy, qz], shape [7]

    Returns:
        torch.Tensor: Transformed point, shape [3]
    """
    position = transform[:3]
    rotation = transform[3:]
    return rotate_vector(point, rotation) + position


def transform_points_batch(
    points: torch.Tensor, transforms: torch.Tensor
) -> torch.Tensor:
    """
    Apply rigid transforms to points in batch.

    Args:
        points (torch.Tensor): Points to transform, shape [..., 3, 1]
        transforms (torch.Tensor): Transforms [x, y, z, qw, qx, qy, qz], shape [..., 7, 1]

    Returns:
        torch.Tensor: Transformed points, shape [..., 3]
    """
    positions = transforms[..., :3, :]
    rotations = transforms[..., 3:, :]
    return rotate_vectors_batch(points, rotations) + positions


def rotvec_to_quat(rotation_vector: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation vector to quaternion.
    rotation_vector = angle * axis, where axis is normalized.
    quaternion = [w, x, y, z]

    Args:
        rotation_vector (torch.Tensor): Rotation vector, shape [3]
    Returns:
        torch.Tensor: Quaternion, shape [4]
    """
    angle = torch.norm(rotation_vector)

    # Handle zero rotation
    if angle < 1e-8:
        return torch.tensor([1.0, 0.0, 0.0, 0.0]).type(rotation_vector.dtype)

    axis = rotation_vector / angle
    half_angle = angle / 2
    sin_half = torch.sin(half_angle)

    w = torch.cos(half_angle)
    x = axis[0] * sin_half
    y = axis[1] * sin_half
    z = axis[2] * sin_half

    return torch.tensor([w, x, y, z]).type(rotation_vector.dtype)


def rotvec_to_quat_batch(rotation_vectors: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation vectors to quaternions in batch.
    rotation_vector = angle * axis, where axis is normalized.
    quaternion = [w, x, y, z]

    Args:
        rotation_vectors (torch.Tensor): Rotation vectors, shape [..., 3]
    Returns:
        torch.Tensor: Quaternions, shape [..., 4]
    """
    angle = torch.norm(rotation_vectors, dim=-1, keepdim=True)

    # Create result tensor
    quats = torch.zeros(
        (*rotation_vectors.shape[:-1], 4), device=rotation_vectors.device
    )

    # Identify zero and non-zero rotations
    mask = angle.squeeze(-1) > 1e-8

    # Handle non-zero rotations
    if mask.any():
        angle_masked = angle[mask]
        axis = rotation_vectors[mask] / angle_masked
        half_angle = angle_masked / 2
        sin_half = torch.sin(half_angle)

        quats[mask, 0] = torch.cos(half_angle).squeeze(-1)
        quats[mask, 1:] = axis * sin_half

    # Handle zero rotations (set to identity quaternion)
    quats[~mask, 0] = 1.0

    return quats.type(rotation_vectors.dtype)


def quat_to_rotvec(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation vector.
    quaternion = [w, x, y, z]
    rotation_vector = angle * axis
    """
    # Normalize quaternion
    quaternion = quaternion / torch.norm(quaternion)

    # Ensure w is positive to match scipy convention
    if quaternion[0] < 0:
        quaternion = -quaternion

    # Extract w component and vector part
    w = quaternion[0]
    xyz = quaternion[1:]

    # Calculate angle
    angle = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))

    # Handle small angles
    if angle < 1e-8:
        return xyz * 2.0

    # Calculate axis
    sin_half_angle = torch.sqrt(1 - w * w)
    axis = xyz / sin_half_angle

    return (angle * axis).type(quaternion.dtype)


def quat_to_rotvec_batch(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation vectors in batch.
    quaternion = [w, x, y, z]
    rotation_vector = angle * axis

    Args:
        quaternions (torch.Tensor): Quaternions, shape [..., 4]
    Returns:
        torch.Tensor: Rotation vectors, shape [..., 3]
    """
    # Normalize quaternions
    quaternions = quaternions / torch.norm(quaternions, dim=-1, keepdim=True)

    # Ensure w is positive (to match scipy convention) for all quaternions
    w_sign = torch.sign(quaternions[..., 0])
    w_sign = torch.where(
        w_sign == 0, torch.tensor(1.0, device=quaternions.device), w_sign
    )
    quaternions = quaternions * w_sign.unsqueeze(-1)

    # Extract w component and vector part
    w = quaternions[..., 0]
    xyz = quaternions[..., 1:]

    # Calculate angle
    angle = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))

    # Create rotation vectors
    rotvecs = torch.zeros_like(xyz)

    # Handle non-zero angles
    non_zero_mask = angle > 1e-8

    if non_zero_mask.any():
        # Calculate axis for non-zero angles
        sin_half_angle = torch.sqrt(1 - w[non_zero_mask] ** 2 + 1e-8)
        axis = xyz[non_zero_mask] / sin_half_angle.unsqueeze(-1)

        # Apply angle * axis for non-zero angles
        rotvecs[non_zero_mask] = axis * angle[non_zero_mask].unsqueeze(-1)

    # Handle small angles
    small_angle_mask = ~non_zero_mask
    if small_angle_mask.any():
        rotvecs[small_angle_mask] = xyz[small_angle_mask] * 2.0

    return rotvecs.type(quaternions.dtype)


def quat_to_rotmat(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.
    quaternion = [w, x, y, z]

    Args:
        quaternion (torch.Tensor): Quaternion, shape [4]
    Returns:
        torch.Tensor: Rotation matrix, shape [3, 3]
    """
    # Normalize quaternion
    quaternion = quaternion / torch.norm(quaternion)

    w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]

    # Calculate rotation matrix
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return torch.tensor(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]
    ).type(quaternion.dtype)


def quat_to_rotmat_batch(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation matrices in batch.
    quaternion = [w, x, y, z]

    Args:
        quaternions (torch.Tensor): Quaternions, shape [..., 4]
    Returns:
        torch.Tensor: Rotation matrices, shape [..., 3, 3]
    """
    # Normalize quaternions
    quaternions = quaternions / torch.norm(quaternions, dim=-1, keepdim=True)

    w, x, y, z = (
        quaternions[..., 0],
        quaternions[..., 1],
        quaternions[..., 2],
        quaternions[..., 3],
    )

    # Calculate components
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # Build rotation matrices
    r00 = 1 - 2 * (yy + zz)
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)

    r10 = 2 * (xy + wz)
    r11 = 1 - 2 * (xx + zz)
    r12 = 2 * (yz - wx)

    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = 1 - 2 * (xx + yy)

    # Stack to form rotation matrices
    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)

    rotation_matrices = torch.stack([row0, row1, row2], dim=-2)

    return rotation_matrices.type(quaternions.dtype)


def rotmat_to_quat(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion.
    quaternion = [w, x, y, z]

    Args:
        rotation_matrix (torch.Tensor): Rotation matrix, shape [3, 3]
    Returns:
        torch.Tensor: Quaternion, shape [4]
    """
    m = rotation_matrix

    # Check if matrix is a proper rotation matrix
    det = torch.det(m)
    if torch.abs(det - 1.0) > 1e-6:
        raise ValueError(
            "Input matrix is not a valid rotation matrix. Determinant is not 1."
        )

    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return torch.tensor([w, x, y, z]).type(rotation_matrix.dtype)


def rotmat_to_quat_batch(rotation_matrices: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions in batch.
    quaternion = [w, x, y, z]

    Args:
        rotation_matrices (torch.Tensor): Rotation matrices, shape [..., 3, 3]
    Returns:
        torch.Tensor: Quaternions, shape [..., 4]
    """
    batch_shape = rotation_matrices.shape[:-2]
    quats = torch.zeros(
        (*batch_shape, 4),
        dtype=rotation_matrices.dtype,
        device=rotation_matrices.device,
    )

    # Extract matrix elements
    m00 = rotation_matrices[..., 0, 0]
    m01 = rotation_matrices[..., 0, 1]
    m02 = rotation_matrices[..., 0, 2]
    m10 = rotation_matrices[..., 1, 0]
    m11 = rotation_matrices[..., 1, 1]
    m12 = rotation_matrices[..., 1, 2]
    m20 = rotation_matrices[..., 2, 0]
    m21 = rotation_matrices[..., 2, 1]
    m22 = rotation_matrices[..., 2, 2]

    # Compute trace
    trace = m00 + m11 + m22

    # Case 1: Trace is positive
    mask_1 = trace > 0
    if mask_1.any():
        s = 0.5 / torch.sqrt(trace[mask_1] + 1.0)
        quats[mask_1, 0] = 0.25 / s
        quats[mask_1, 1] = (m21[mask_1] - m12[mask_1]) * s
        quats[mask_1, 2] = (m02[mask_1] - m20[mask_1]) * s
        quats[mask_1, 3] = (m10[mask_1] - m01[mask_1]) * s

    # Case 2: m00 is largest diagonal element
    mask_2 = (~mask_1) & (m00 > m11) & (m00 > m22)
    if mask_2.any():
        s = 2.0 * torch.sqrt(1.0 + m00[mask_2] - m11[mask_2] - m22[mask_2])
        quats[mask_2, 0] = (m21[mask_2] - m12[mask_2]) / s
        quats[mask_2, 1] = 0.25 * s
        quats[mask_2, 2] = (m01[mask_2] + m10[mask_2]) / s
        quats[mask_2, 3] = (m02[mask_2] + m20[mask_2]) / s

    # Case 3: m11 is largest diagonal element
    mask_3 = (~mask_1) & (~mask_2) & (m11 > m22)
    if mask_3.any():
        s = 2.0 * torch.sqrt(1.0 + m11[mask_3] - m00[mask_3] - m22[mask_3])
        quats[mask_3, 0] = (m02[mask_3] - m20[mask_3]) / s
        quats[mask_3, 1] = (m01[mask_3] + m10[mask_3]) / s
        quats[mask_3, 2] = 0.25 * s
        quats[mask_3, 3] = (m12[mask_3] + m21[mask_3]) / s

    # Case 4: m22 is largest diagonal element
    mask_4 = (~mask_1) & (~mask_2) & (~mask_3)
    if mask_4.any():
        s = 2.0 * torch.sqrt(1.0 + m22[mask_4] - m00[mask_4] - m11[mask_4])
        quats[mask_4, 0] = (m10[mask_4] - m01[mask_4]) / s
        quats[mask_4, 1] = (m02[mask_4] + m20[mask_4]) / s
        quats[mask_4, 2] = (m12[mask_4] + m21[mask_4]) / s
        quats[mask_4, 3] = 0.25 * s

    # Normalize quaternions
    quats = quats / torch.norm(quats, dim=-1, keepdim=True)

    return quats


def relative_rotation(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
    """
    Compute the relative rotation between two poses.
    This rotation can take vectors in the frame of q_a and transform them to the frame of q_b.

    Args:
        q_a (torch.Tensor): First quaternion, shape [4]
        q_b (torch.Tensor): Second quaternion, shape [4]

    Returns:
        torch.Tensor: Relative rotation quaternion, shape [4]
    """
    return quat_mul(q_b, quat_inv(q_a))


def relative_rotation_batch(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
    """
    Compute the relative rotation between two poses in batch.
    This rotation can take vectors in the frame of q_a and transform them to the frame of q_b.

    Args:
        q_a (torch.Tensor): First quaternions, shape [..., 4]
        q_b (torch.Tensor): Second quaternions, shape [..., 4]

    Returns:
        torch.Tensor: Relative rotation quaternions, shape [..., 4]
    """
    return quat_mul_batch(q_b, quat_inv_batch(q_a))


def relative_transform(body_q_a: torch.Tensor, body_q_b: torch.Tensor) -> torch.Tensor:
    """
    Compute the relative transform between two poses.
    This transform can take vectors in the frame of q_a and transform them to the frame of q_b.

    Args:
        body_q_a (torch.Tensor): First transform [x, y, z, qw, qx, qy, qz], shape [7]
        body_q_b (torch.Tensor): Second transform [x, y, z, qw, qx, qy, qz], shape [7]

    Returns:
        torch.Tensor: Relative transform, shape [7]
    """
    pos_a = body_q_a[:3]
    pos_b = body_q_b[:3]
    quat_a = body_q_a[3:]
    quat_b = body_q_b[3:]

    # Translation: rotate (pos_a - pos_b) by inverse of quat_b
    t = rotate_vector_inverse(pos_a - pos_b, quat_b)

    # Rotation: quat_b * quat_a^-1
    q = quat_mul(quat_b, quat_inv(quat_a))

    return torch.cat([t, q])


def relative_transform(body_q_a: torch.Tensor, body_q_b: torch.Tensor) -> torch.Tensor:
    """
    Compute the relative transform from a to b.
    This transform can take vectors in the frame of a and transform them to the frame of b.

    Args:
        body_q_a (torch.Tensor): First transform [x, y, z, qw, qx, qy, qz], shape [7]
        body_q_b (torch.Tensor): Second transform [x, y, z, qw, qx, qy, qz], shape [7]

    Returns:
        torch.Tensor: Relative transform, shape [7]
    """
    pos_a = body_q_a[:3]
    pos_b = body_q_b[:3]
    quat_a = body_q_a[3:]
    quat_b = body_q_b[3:]

    # Translation: first express (pos_a - pos_b) in the frame of b
    t = rotate_vector_inverse(pos_a - pos_b, quat_b)

    # Rotation: inverse of quat_a composed with quat_b
    q = quat_mul(quat_inv(quat_a), quat_b)

    return torch.cat([t, q])


def relative_transform_batch(
    body_q_a: torch.Tensor, body_q_b: torch.Tensor
) -> torch.Tensor:
    """
    Compute the relative transform between two poses in batch.
    This transform can take vectors in the frame of q_a and transform them to the frame of q_b.

    Args:
        body_q_a (torch.Tensor): First transforms [x, y, z, qw, qx, qy, qz], shape [..., 7]
        body_q_b (torch.Tensor): Second transforms [x, y, z, qw, qx, qy, qz], shape [..., 7]

    Returns:
        torch.Tensor: Relative transforms, shape [..., 7]
    """
    pos_a = body_q_a[..., :3]
    pos_b = body_q_b[..., :3]
    quat_a = body_q_a[..., 3:]
    quat_b = body_q_b[..., 3:]

    # Translation: rotate (pos_a - pos_b) by inverse of quat_b
    t = rotate_vectors_inverse_batch(pos_a - pos_b, quat_b)

    # Rotation: quat_b * quat_a^-1
    q = quat_mul_batch(quat_b, quat_inv_batch(quat_a))

    return torch.cat([t, q], dim=-1)


def delta_q_fixed(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
    """
    Compute the delta quaternion between two poses.

    Args:
        q_a (torch.Tensor): First quaternion, shape [4]
        q_b (torch.Tensor): Second quaternion, shape [4]

    Returns:
        torch.Tensor: Delta quaternion, shape [4]
    """
    return quat_mul(q_a, quat_inv(q_b))


def delta_q_fixed_batch(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
    """
    Compute the delta quaternion between two poses in batch.

    Args:
        q_a (torch.Tensor): First quaternions, shape [..., 4]
        q_b (torch.Tensor): Second quaternions, shape [..., 4]

    Returns:
        torch.Tensor: Delta quaternions, shape [..., 4]
    """
    return quat_mul_batch(q_a, quat_inv_batch(q_b))


def body_to_world(r: torch.Tensor, body_q: torch.Tensor) -> torch.Tensor:
    """
    Convert a point from body frame to world frame.

    Args:
        r (torch.Tensor): Point in body frame, shape [3]
        body_q (torch.Tensor): Body transform [x, y, z, qw, qx, qy, qz], shape [7]

    Returns:
        torch.Tensor: Point in world frame, shape [3]
    """
    return transform_point(r, body_q)


def body_to_world_batch(r: torch.Tensor, body_q: torch.Tensor) -> torch.Tensor:
    """
    Convert points from body frames to world frame in batch.

    Args:
        r (torch.Tensor): Points in body frames, shape [..., 3]
        body_q (torch.Tensor): Body transforms [x, y, z, qw, qx, qy, qz], shape [..., 7]

    Returns:
        torch.Tensor: Points in world frame, shape [..., 3]
    """
    return transform_points_batch(r, body_q)


def world_to_body(r: torch.Tensor, body_q: torch.Tensor) -> torch.Tensor:
    """
    Convert a point from world frame to body frame.

    Args:
        r (torch.Tensor): Point in world frame, shape [3]
        body_q (torch.Tensor): Body transform [x, y, z, qw, qx, qy, qz], shape [7]

    Returns:
        torch.Tensor: Point in body frame, shape [3]
    """
    body_x = body_q[:3]
    body_rot = body_q[3:]
    return rotate_vector_inverse(r - body_x, body_rot)


def world_to_body_batch(r: torch.Tensor, body_q: torch.Tensor) -> torch.Tensor:
    """
    Convert points from world frame to body frames in batch.

    Args:
        r (torch.Tensor): Points in world frame, shape [..., 3]
        body_q (torch.Tensor): Body transforms [x, y, z, qw, qx, qy, qz], shape [..., 7]

    Returns:
        torch.Tensor: Points in body frames, shape [..., 3]
    """
    body_x = body_q[..., :3]
    body_rot = body_q[..., 3:]
    return rotate_vectors_inverse_batch(r - body_x, body_rot)


def transform_multiply(
    transform_a: torch.Tensor, transform_b: torch.Tensor
) -> torch.Tensor:
    """
    Multiply two transforms (compose transformations).

    Args:
        transform_a (torch.Tensor): First transform [x, y, z, qw, qx, qy, qz], shape [7]
        transform_b (torch.Tensor): Second transform [x, y, z, qw, qx, qy, qz], shape [7]

    Returns:
        torch.Tensor: Composed transform, shape [7]
    """
    # Extract positions and orientations
    x_a = transform_a[:3]  # position of frame A
    q_a = transform_a[3:]  # orientation of frame A
    x_b = transform_b[:3]  # position of frame B
    q_b = transform_b[3:]  # orientation of frame B

    # First combine the rotations
    q = quat_mul(q_a, q_b)

    # Then transform the position:
    # 1. Rotate frame B's position by frame A's rotation
    # 2. Add frame A's position
    x = x_a + rotate_vector(x_b, q_a)

    return torch.cat([x, q])


def transform_multiply_batch(
    transform_a: torch.Tensor, transform_b: torch.Tensor
) -> torch.Tensor:
    """
    Multiply transforms in batch (compose transformations).

    Args:
        transform_a (torch.Tensor): First transforms [x, y, z, qw, qx, qy, qz], shape [N, 7, 1]
        transform_b (torch.Tensor): Second transforms [x, y, z, qw, qx, qy, qz], shape [N, 7, 1]

    Returns:
        torch.Tensor: Composed transforms, shape [N, 7, 1]
    """
    # Extract positions and orientations
    x_a = transform_a[:, :3]  # position of frame A
    q_a = transform_a[:, 3:]  # orientation of frame A
    x_b = transform_b[:, :3]  # position of frame B
    q_b = transform_b[:, 3:]  # orientation of frame B

    # Combine the rotations
    q = quat_mul_batch(q_a, q_b)

    # Transform the position
    x = x_a + rotate_vectors_batch(x_b, q_a)

    return torch.cat([x, q], dim=1)
