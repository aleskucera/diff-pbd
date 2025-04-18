import time
from typing import Tuple

from demos.utils import *
from pbd_torch.constants import *
from pbd_torch.transform import *


def positional_delta(
    body_q_a: torch.Tensor,
    body_q_b: torch.Tensor,
    r_a: torch.Tensor,
    r_b: torch.Tensor,
    m_a_inv: torch.Tensor,
    m_b_inv: torch.Tensor,
    I_a_inv: torch.Tensor,
    I_b_inv: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dbody_q_a = torch.zeros(
        7, device=body_q_a.device
    )  # Linear and angular correction for body A
    dbody_q_b = torch.zeros(
        7, device=body_q_a.device
    )  # Linear and angular correction for body B
    d_lambda = torch.tensor([0.0], device=body_q_a.device)  # Impulse magnitude

    q_a = body_q_a[3:]  # Orientation of body A
    q_b = body_q_b[3:]  # Orientation of body B

    # Compute the relative position of the contact point in the world frame
    delta_x = transform_point(r_a, body_q_a) - transform_point(r_b, body_q_b)

    c = torch.norm(delta_x)
    if c < 1e-6:
        return dbody_q_a, dbody_q_b, d_lambda

    n = delta_x / c  # Normal vector pointing from body B to body A in the world frame
    n_a = rotate_vector_inverse(n, q_a)  # Rotate the normal to the body A frame
    n_b = rotate_vector_inverse(n, q_b)  # Rotate the normal to the body B frame

    weight_a = m_a_inv + torch.dot(
        torch.linalg.cross(I_a_inv @ torch.linalg.cross(r_a, n_a), r_a), n_a
    )
    weight_b = m_b_inv + torch.dot(
        torch.linalg.cross(I_b_inv @ torch.linalg.cross(r_b, n_b), r_b), n_b
    )

    # Compute the impulse magnitude
    d_lambda = -c / (weight_a + weight_b)

    # Compute the impulse vector
    p = d_lambda * n  # Impulse in the world frame
    p_a = d_lambda * n_a  # Impulse in the body A frame
    p_b = d_lambda * n_b  # Impulse in the body B frame

    # Positional correction
    dx_a = p * m_a_inv
    dx_b = -p * m_b_inv

    # Rotational correction
    w_a = I_a_inv @ torch.linalg.cross(
        r_a, p_a
    )  # Angular velocity correction for body A
    w_a_quat = torch.cat([torch.tensor([0.0], device=w_a.device), w_a], dim=0)
    dq_a = 0.5 * quat_mul(q_a, w_a_quat)

    w_b = I_b_inv @ torch.linalg.cross(
        r_b, p_b
    )  # Angular velocity correction for body B
    w_b_quat = torch.cat([torch.tensor([0.0], device=w_b.device), w_b], dim=0)
    dq_b = -0.5 * quat_mul(q_b, w_b_quat)

    dbody_q_a = torch.cat([dx_a, dq_a])
    dbody_q_b = torch.cat([dx_b, dq_b])

    return dbody_q_a, dbody_q_b, d_lambda


def positional_deltas_batch(
    body_trans_a: torch.Tensor,  # [B, 7, 1]
    body_trans_b: torch.Tensor,  # [B, 7, 1]
    r_a: torch.Tensor,  # [B, 3, 1]
    r_b: torch.Tensor,  # [B, 3, 1]
    m_a_inv: torch.Tensor,  # [B, 1, 1]
    m_b_inv: torch.Tensor,  # [B, 1, 1]
    I_a_inv: torch.Tensor,  # [B, 3, 3]
    I_b_inv: torch.Tensor,  # [B, 3, 3]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = body_trans_a.shape[0]
    device = body_trans_a.device

    # Initialize outputs
    dbody_trans_a = torch.zeros((batch_size, 7, 1), device=device)
    dbody_trans_b = torch.zeros((batch_size, 7, 1), device=device)
    d_lambda = torch.zeros((batch_size, 1), device=device)

    # Extract orientations
    q_a = body_trans_a[:, 3:]  # [B, 4, 1]
    q_b = body_trans_b[:, 3:]  # [B, 4, 1]

    # Compute the contact points in world space
    r_a_world = transform_points_batch(r_a, body_trans_a)  # [B, 3, 1]
    r_b_world = transform_points_batch(r_b, body_trans_b)  # [B, 3, 1]

    # Compute the relative position
    delta_x = r_a_world - r_b_world  # [B, 3, 1]

    # Compute constraint magnitude
    c = torch.norm(delta_x, dim=1, keepdim=True)  # [B, 1, 1]

    # Compute normal vectors
    n = delta_x / (c + 1e-6)  # [B, 3, 1]

    # Rotate normals to body frames
    n_a = rotate_vectors_inverse_batch(n, q_a)  # [B, 3, 1]
    n_b = rotate_vectors_inverse_batch(n, q_b)  # [B, 3, 1]

    # Compute crosses and dots for weights
    r_cross_n_a = torch.linalg.cross(r_a, n_a, dim=1)  # [B, 3, 1]
    r_cross_n_b = torch.linalg.cross(r_b, n_b, dim=1)  # [B, 3, 1]

    # Compute I_inv @ cross products
    I_inv_cross_a = torch.matmul(I_a_inv, r_cross_n_a)  # [B, 3, 1]
    I_inv_cross_b = torch.matmul(I_b_inv, r_cross_n_b)  # [B, 3, 1]

    # Cross products with r
    final_cross_a = torch.cross(I_inv_cross_a, r_a, dim=1)  # [B, 3, 1]
    final_cross_b = torch.cross(I_inv_cross_b, r_b, dim=1)  # [B, 3, 1]

    # Dot products for weights
    dot_a = torch.sum(final_cross_a * n_a, dim=1)  # [B, 1]
    dot_b = torch.sum(final_cross_b * n_b, dim=1)  # [B, 1]

    # Compute weights
    weight_a = m_a_inv.squeeze(1) + dot_a  # [batch_size]
    weight_b = m_b_inv.squeeze(1) + dot_b  # [batch_size]

    # Compute the impulse magnitude
    d_lambda = -c / (weight_a + weight_b)

    # Compute impulse vectors
    p = d_lambda.unsqueeze(1) * n  # [batch_size, 3]
    p_a = d_lambda.unsqueeze(1) * n_a  # [batch_size, 3]
    p_b = d_lambda.unsqueeze(1) * n_b  # [batch_size, 3]

    # Positional corrections
    dx_a = p * m_a_inv  # [batch_size, 3]
    dx_b = -p * m_b_inv  # [batch_size, 3]

    # Rotational corrections
    r_cross_p_a = torch.cross(r_a, p_a, dim=1)  # [batch_size, 3]
    r_cross_p_b = torch.cross(r_b, p_b, dim=1)  # [batch_size, 3]

    w_a = torch.bmm(I_a_inv, r_cross_p_a.unsqueeze(2)).squeeze(2)  # [batch_size, 3]
    w_b = torch.bmm(I_b_inv, r_cross_p_b.unsqueeze(2)).squeeze(2)  # [batch_size, 3]

    # Create quaternion vectors for rotational corrections
    zeros = torch.zeros(batch_size, 1, device=device)
    w_a_quat = torch.cat([zeros, w_a], dim=1)  # [batch_size, 4]
    w_b_quat = torch.cat([zeros, w_b], dim=1)  # [batch_size, 4]

    # Quaternion multiplication
    dq_a = 0.5 * quat_mul_batch(q_a, w_a_quat)  # [batch_size, 4]
    dq_b = -0.5 * quat_mul_batch(q_b, w_b_quat)  # [batch_size, 4]

    # Combine positional and rotational corrections
    dbody_trans_a = torch.cat([dx_a, dq_a], dim=1)  # [batch_size, 7]
    dbody_trans_b = torch.cat([dx_b, dq_b], dim=1)  # [batch_size, 7]

    return dbody_trans_a, dbody_trans_b, d_lambda


def joint_angular_delta(
    body_q_p: torch.Tensor,
    body_q_c: torch.Tensor,
    X_p: torch.Tensor,
    X_c: torch.Tensor,
    joint_axis: torch.Tensor,
    I_p_inv: torch.Tensor,
    I_c_inv: torch.Tensor,
):
    q_p = body_q_p[3:]
    q_c = body_q_c[3:]

    X_wj_p = transform_multiply(body_q_p, X_p)
    X_wj_c = transform_multiply(body_q_c, X_c)

    axis_w_p = rotate_vector(joint_axis, X_wj_p[3:])
    axis_w_c = rotate_vector(joint_axis, X_wj_c[3:])

    rot_vector = torch.linalg.cross(axis_w_p, axis_w_c)
    theta = torch.linalg.norm(rot_vector)

    if theta < 1e-6:
        return torch.zeros(4), torch.zeros(4)

    n = rot_vector / theta
    n_p = rotate_vector_inverse(n, q_p)
    n_c = rotate_vector_inverse(n, q_c)

    weight_p = n_p @ I_p_inv @ n_p
    weight_c = n_c @ I_c_inv @ n_c
    weight = weight_p + weight_c

    theta_p = theta * weight_p / weight
    theta_c = -theta * weight_c / weight

    w_p_quat = torch.cat([torch.tensor([0.0]), theta_p * n], dim=0)
    w_c_quat = torch.cat([torch.tensor([0.0]), theta_c * n], dim=0)

    dq_p = 0.5 * quat_mul(w_p_quat, q_p)
    dq_c = 0.5 * quat_mul(w_c_quat, q_c)

    return dq_p, dq_c


def joint_angular_deltas_batch(
    body_q_p: torch.Tensor,  # [batch_size, 7]
    body_q_c: torch.Tensor,  # [batch_size, 7]
    X_p: torch.Tensor,  # [batch_size, 7]
    X_c: torch.Tensor,  # [batch_size, 7]
    joint_axis: torch.Tensor,  # [batch_size, 3]
    I_p_inv: torch.Tensor,  # [batch_size, 3, 3]
    I_c_inv: torch.Tensor,  # [batch_size, 3, 3]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched version of joint_angular_delta that processes multiple joints at once."""
    batch_size = body_q_p.shape[0]
    device = body_q_p.device

    # Extract orientations
    q_p = body_q_p[:, 3:]  # [batch_size, 4]
    q_c = body_q_c[:, 3:]  # [batch_size, 4]

    # Transform joint frames to world space
    X_wj_p = transform_multiply_batch(body_q_p, X_p)  # [batch_size, 7]
    X_wj_c = transform_multiply_batch(body_q_c, X_c)  # [batch_size, 7]

    # Rotate joint axes to world space
    axis_w_p = rotate_vectors_batch(joint_axis, X_wj_p[:, 3:])  # [batch_size, 3]
    axis_w_c = rotate_vectors_batch(joint_axis, X_wj_c[:, 3:])  # [batch_size, 3]

    # Compute rotation vector and angle
    rot_vector = torch.cross(axis_w_p, axis_w_c, dim=1)  # [batch_size, 3]
    theta = torch.norm(rot_vector, dim=1)  # [batch_size]

    # Initialize outputs
    dq_p = torch.zeros((batch_size, 4), device=device)
    dq_c = torch.zeros((batch_size, 4), device=device)

    # Create mask for non-zero thetas to avoid division by zero
    mask = theta > 1e-6

    # Process only the joints with significant angles
    n = torch.zeros_like(rot_vector)
    n[mask] = rot_vector[mask] / theta[mask].unsqueeze(1)  # [batch_size, 3]

    # Rotate normals to body frames
    n_p = rotate_vectors_inverse_batch(n, q_p)  # [batch_size, 3]
    n_c = rotate_vectors_inverse_batch(n, q_c)  # [batch_size, 3]

    # Compute weights using matrix-vector multiplications
    # I_inv @ n
    I_p_inv_n = torch.bmm(I_p_inv, n_p.unsqueeze(2)).squeeze(2)  # [batch_size, 3]
    I_c_inv_n = torch.bmm(I_c_inv, n_c.unsqueeze(2)).squeeze(2)  # [batch_size, 3]

    # Dot products for weights: n_p @ I_p_inv @ n_p and n_c @ I_c_inv @ n_c
    weight_p = torch.sum(n_p * I_p_inv_n, dim=1)  # [batch_size]
    weight_c = torch.sum(n_c * I_c_inv_n, dim=1)  # [batch_size]

    weight = weight_p + weight_c  # [batch_size]

    # Compute angle distributions
    theta_p = torch.zeros_like(theta)
    theta_c = torch.zeros_like(theta)

    valid_weights = mask & (weight > 1e-6)
    theta_p[valid_weights] = (
        theta[valid_weights] * weight_p[valid_weights] / weight[valid_weights]
    )
    theta_c[valid_weights] = (
        -theta[valid_weights] * weight_c[valid_weights] / weight[valid_weights]
    )

    # Create quaternion vectors for rotational corrections
    zeros = torch.zeros(batch_size, 1, device=device)

    # Multiply n by theta_p and theta_c
    n_p_scaled = n * theta_p.unsqueeze(1)  # [batch_size, 3]
    n_c_scaled = n * theta_c.unsqueeze(1)  # [batch_size, 3]

    w_p_quat = torch.cat([zeros, n_p_scaled], dim=1)  # [batch_size, 4]
    w_c_quat = torch.cat([zeros, n_c_scaled], dim=1)  # [batch_size, 4]

    # Quaternion multiplication
    dq_p = 0.5 * quat_mul_batch(w_p_quat, q_p)  # [batch_size, 4]
    dq_c = 0.5 * quat_mul_batch(w_c_quat, q_c)  # [batch_size, 4]

    return dq_p, dq_c


def joint_delta(
    body_q_p: torch.Tensor,
    body_q_c: torch.Tensor,
    X_p: torch.Tensor,
    X_c: torch.Tensor,
    joint_axis: torch.Tensor,
    m_p_inv: torch.Tensor,
    m_c_inv: torch.Tensor,
    I_p_inv: torch.Tensor,
    I_c_inv: torch.Tensor,
):

    dq_p, dq_c = joint_angular_delta(
        body_q_p, body_q_c, X_p, X_c, joint_axis, I_p_inv, I_c_inv
    )

    new_body_q_p = body_q_p.clone()
    new_body_q_c = body_q_c.clone()
    new_body_q_p[3:] = normalize_quat(new_body_q_p[3:] + dq_p)
    new_body_q_c[3:] = normalize_quat(new_body_q_c[3:] + dq_c)

    dbody_q_p, dbody_q_c, _ = positional_delta(
        body_q_a=new_body_q_p,
        body_q_b=new_body_q_c,
        r_a=X_p[:3],
        r_b=X_c[:3],
        m_a_inv=m_p_inv,
        m_b_inv=m_c_inv,
        I_a_inv=I_p_inv,
        I_b_inv=I_c_inv,
    )
    dbody_q_p[3:] = dbody_q_p[3:] + dq_p
    dbody_q_c[3:] = dbody_q_c[3:] + dq_c

    return dbody_q_p, dbody_q_c


def joint_deltas_batch(
    body_q_p: torch.Tensor,  # [batch_size, 7]
    body_q_c: torch.Tensor,  # [batch_size, 7]
    X_p: torch.Tensor,  # [batch_size, 7]
    X_c: torch.Tensor,  # [batch_size, 7]
    joint_axis: torch.Tensor,  # [batch_size, 3]
    m_p_inv: torch.Tensor,  # [batch_size, 1]
    m_c_inv: torch.Tensor,  # [batch_size, 1]
    I_p_inv: torch.Tensor,  # [batch_size, 3, 3]
    I_c_inv: torch.Tensor,  # [batch_size, 3, 3]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched version of joint_delta that processes multiple joints at once."""

    # Get angular corrections
    dq_p, dq_c = joint_angular_deltas_batch(
        body_q_p, body_q_c, X_p, X_c, joint_axis, I_p_inv, I_c_inv
    )

    # Create copies of body positions/orientations
    new_body_q_p = body_q_p.clone()
    new_body_q_c = body_q_c.clone()

    # Apply angular corrections
    new_body_q_p[:, 3:] = normalize_quat_batch(new_body_q_p[:, 3:] + dq_p)
    new_body_q_c[:, 3:] = normalize_quat_batch(new_body_q_c[:, 3:] + dq_c)

    # Get positional corrections
    dbody_q_p, dbody_q_c, _ = positional_deltas_batch(
        body_trans_a=new_body_q_p,
        body_trans_b=new_body_q_c,
        r_a=X_p[:, :3],
        r_b=X_c[:, :3],
        m_a_inv=m_p_inv,
        m_b_inv=m_c_inv,
        I_a_inv=I_p_inv,
        I_b_inv=I_c_inv,
    )

    # Combine angular and positional corrections
    dbody_q_p[:, 3:] = dbody_q_p[:, 3:] + dq_p
    dbody_q_c[:, 3:] = dbody_q_c[:, 3:] + dq_c

    return dbody_q_p, dbody_q_c


def ground_restitution_delta(
    body_q: torch.Tensor,
    body_qd: torch.Tensor,
    body_qd_prev: torch.Tensor,
    r: torch.Tensor,
    n: torch.Tensor,
    m_inv: torch.Tensor,
    I_inv: torch.Tensor,
    restitution: torch.Tensor,
):
    dbody_qd = torch.zeros(6, device=body_q.device)  # Linear and angular correction

    q = body_q[3:]  # Orientation
    v = body_qd[3:]  # Linear velocity
    w = body_qd[:3]  # Angular velocity
    v_prev = body_qd_prev[3:]  # Previous linear velocity
    w_prev = body_qd_prev[:3]  # Previous angular velocity

    # Compute the normal component of the relative velocity
    v_rel = v + rotate_vector(torch.linalg.cross(w, r), q)
    vn = torch.dot(v_rel, n)

    # Compute the normal component of the relative velocity before the velocity update
    v_rel_prev = v_prev + rotate_vector(torch.linalg.cross(w_prev, r), q)
    vn_prev = torch.dot(v_rel_prev, n)
    # vn_prev = torch.max(vn_prev, torch.tensor([0.0]))

    velocity_correction = torch.max(
        torch.tensor([0.0], device=vn.device), (vn + restitution * vn_prev)
    )

    # If the correction is too small we can ignore it to avoid jittering
    if torch.abs(velocity_correction) < 2 * 9.81 * 0.01:  # 2 * gravity * dt
        return dbody_qd

    # Compute the change of velocity due to the restitution
    delta_v_restitution = -n * velocity_correction

    # Compute the impulse due to the restitution in the world frame
    nb = rotate_vector_inverse(n, q)  # Rotate the normal to the body frame
    J_restitution = delta_v_restitution / (
        m_inv + torch.dot(torch.linalg.cross(I_inv @ torch.linalg.cross(r, nb), r), nb)
    )
    dv = J_restitution * m_inv
    dw = I_inv @ torch.linalg.cross(r, rotate_vector_inverse(J_restitution, q))

    dbody_qd = torch.cat([dw, dv])
    return dbody_qd


def ground_restitution_delta_batch(
    body_q: torch.Tensor,  # [batch_size, 7]
    body_qd: torch.Tensor,  # [batch_size, 6]
    body_qd_prev: torch.Tensor,  # [batch_size, 6]
    r: torch.Tensor,  # [batch_size, 3]
    n: torch.Tensor,  # [batch_size, 3]
    m_inv: torch.Tensor,  # [batch_size, 1]
    I_inv: torch.Tensor,  # [batch_size, 3, 3]
    restitution: torch.Tensor,  # [batch_size, 1] or [batch_size]
):
    """Batched version of ground_restitution_delta that processes multiple contacts at once."""
    batch_size = body_q.shape[0]
    device = body_q.device

    # Initialize outputs
    dbody_qd = torch.zeros((batch_size, 6), device=device)

    # Extract orientations and velocities
    q = body_q[:, 3:]  # Orientation [batch_size, 4]
    v = body_qd[:, 3:]  # Linear velocity [batch_size, 3]
    w = body_qd[:, :3]  # Angular velocity [batch_size, 3]
    v_prev = body_qd_prev[:, 3:]  # Previous linear velocity [batch_size, 3]
    w_prev = body_qd_prev[:, :3]  # Previous angular velocity [batch_size, 3]

    # Compute cross products of angular velocity and contact point
    w_cross_r = torch.cross(w, r, dim=1)  # [batch_size, 3]
    w_prev_cross_r = torch.cross(w_prev, r, dim=1)  # [batch_size, 3]

    # Rotate to world space
    w_cross_r_world = rotate_vectors_batch(w_cross_r, q)  # [batch_size, 3]
    w_prev_cross_r_world = rotate_vectors_batch(w_prev_cross_r, q)  # [batch_size, 3]

    # Compute the relative velocities
    v_rel = v + w_cross_r_world  # [batch_size, 3]
    v_rel_prev = v_prev + w_prev_cross_r_world  # [batch_size, 3]

    # Compute normal components of relative velocities
    vn = torch.sum(v_rel * n, dim=1, keepdim=True)  # [batch_size, 1]
    vn_prev = torch.sum(v_rel_prev * n, dim=1, keepdim=True)  # [batch_size, 1]

    # Ensure restitution has correct shape
    if restitution.dim() == 1:
        restitution = restitution.unsqueeze(1)  # [batch_size, 1]

    # Compute velocity corrections
    zeros = torch.zeros_like(vn)
    velocity_correction = torch.max(
        zeros, vn + restitution * vn_prev
    )  # [batch_size, 1]

    # Apply threshold to avoid jittering (2 * gravity * dt)
    jitter_threshold = torch.tensor(2 * 9.81 * 0.01, device=device)
    mask = torch.abs(velocity_correction) >= jitter_threshold

    if not mask.any():
        return dbody_qd

    # Compute the impulse direction (negative normal)
    delta_v_restitution = -n * velocity_correction  # [batch_size, 3]

    # Rotate normals to body frames
    nb = rotate_vectors_inverse_batch(n, q)  # [batch_size, 3]

    # Compute denominator for impulse calculation
    r_cross_nb = torch.cross(r, nb, dim=1)  # [batch_size, 3]
    I_inv_cross = torch.bmm(I_inv, r_cross_nb.unsqueeze(2)).squeeze(
        2
    )  # [batch_size, 3]
    r_cross_I_inv_cross = torch.cross(I_inv_cross, r, dim=1)  # [batch_size, 3]
    dot_products = torch.sum(
        r_cross_I_inv_cross * nb, dim=1, keepdim=True
    )  # [batch_size, 1]
    denominator = m_inv + dot_products  # [batch_size, 1]

    # Compute impulses
    J_restitution = delta_v_restitution / denominator  # [batch_size, 3]

    # Compute velocity changes
    dv = J_restitution * m_inv  # [batch_size, 3]
    J_body = rotate_vectors_inverse_batch(J_restitution, q)  # [batch_size, 3]
    r_cross_J = torch.cross(r, J_body, dim=1)  # [batch_size, 3]
    dw = torch.bmm(I_inv, r_cross_J.unsqueeze(2)).squeeze(2)  # [batch_size, 3]

    # Combine angular and linear velocity changes
    dbody_qd = torch.cat([dw, dv], dim=1)  # [batch_size, 6]

    # Only apply corrections to contacts that exceeded the jitter threshold
    dbody_qd = dbody_qd * mask.repeat(1, 6)

    return dbody_qd


def ground_dynamic_friction_delta(
    body_q: torch.Tensor,
    body_qd: torch.Tensor,
    r: torch.Tensor,
    n: torch.Tensor,
    m_inv: torch.Tensor,
    I_inv: torch.Tensor,
    dynamic_friction: torch.Tensor,
    lambda_n: torch.Tensor,
    dt: float,
):
    dbody_qd = torch.zeros(6, device=body_q.device)  # Linear and angular correction

    q = body_q[3:]  # Orientation
    v = body_qd[3:]  # Linear velocity
    w = body_qd[:3]  # Angular velocity

    # Compute the relative velocity of the contact point in the world frame
    v_rel = v + rotate_vector(torch.linalg.cross(w, r), q)

    # Compute the tangent component of the relative velocity
    v_t = v_rel - torch.dot(v_rel, n) * n
    v_t_magnitude = torch.norm(v_t)
    t = v_t / v_t_magnitude

    if v_t_magnitude < 1e-5:
        return dbody_qd

    # Compute the change of velocity due to the friction (now infinite friction)
    coulomb_friction = torch.abs(dynamic_friction * lambda_n / (dt**2))
    delta_v_friction = -v_t * torch.min(coulomb_friction, v_t_magnitude) / v_t_magnitude

    # Compute the impulse due to the friction in the world frame
    tb = rotate_vector_inverse(t, q)  # Rotate the tangent to the body frame
    J_friction = delta_v_friction / (
        m_inv + torch.dot(torch.linalg.cross(I_inv @ torch.linalg.cross(r, tb), r), tb)
    )
    dv = J_friction * m_inv
    dw = I_inv @ torch.linalg.cross(r, rotate_vector_inverse(J_friction, q))

    dbody_qd = torch.cat([dw, dv])
    return dbody_qd


def ground_dynamic_friction_delta_batch(
    body_q: torch.Tensor,  # [batch_size, 7]
    body_qd: torch.Tensor,  # [batch_size, 6]
    r: torch.Tensor,  # [batch_size, 3]
    n: torch.Tensor,  # [batch_size, 3]
    m_inv: torch.Tensor,  # [batch_size, 1]
    I_inv: torch.Tensor,  # [batch_size, 3, 3]
    dynamic_friction: torch.Tensor,  # [batch_size, 1] or [batch_size]
    lambda_n: torch.Tensor,  # [batch_size, 1] or [batch_size]
    dt: float,
):
    """Batched version of ground_dynamic_friction_delta that processes multiple contacts at once."""
    batch_size = body_q.shape[0]
    device = body_q.device

    # Initialize outputs
    dbody_qd = torch.zeros((batch_size, 6), device=device)

    # Extract orientations and velocities
    q = body_q[:, 3:]  # Orientation [batch_size, 4]
    v = body_qd[:, 3:]  # Linear velocity [batch_size, 3]
    w = body_qd[:, :3]  # Angular velocity [batch_size, 3]

    # Compute cross products of angular velocity and contact point
    w_cross_r = torch.cross(w, r, dim=1)  # [batch_size, 3]

    # Rotate to world space
    w_cross_r_world = rotate_vectors_batch(w_cross_r, q)  # [batch_size, 3]

    # Compute the relative velocities
    v_rel = v + w_cross_r_world  # [batch_size, 3]

    # Compute normal component and project to get tangential component
    vn_components = torch.sum(v_rel * n, dim=1, keepdim=True) * n  # [batch_size, 3]
    v_t = v_rel - vn_components  # [batch_size, 3]

    # Compute tangential velocity magnitude
    v_t_magnitude = torch.norm(v_t, dim=1, keepdim=True)  # [batch_size, 1]

    # Create mask for non-zero tangential velocities
    mask = v_t_magnitude >= 1e-5

    if not mask.any():
        return dbody_qd

    # Compute normalized tangent direction
    t = torch.zeros_like(v_t)
    t[mask.repeat(1, 3)] = v_t[mask.repeat(1, 3)] / v_t_magnitude[mask].repeat(1, 3)

    # Ensure dynamic_friction and lambda_n have correct shape
    if dynamic_friction.dim() == 1:
        dynamic_friction = dynamic_friction.unsqueeze(1)  # [batch_size, 1]
    if lambda_n.dim() == 1:
        lambda_n = lambda_n.unsqueeze(1)  # [batch_size, 1]

    # Compute Coulomb friction
    coulomb_friction = torch.abs(
        dynamic_friction * lambda_n / (dt**2)
    )  # [batch_size, 1]

    # Compute velocity change due to friction
    min_values = torch.min(coulomb_friction, v_t_magnitude)  # [batch_size, 1]
    scale_factor = min_values / v_t_magnitude  # [batch_size, 1]
    delta_v_friction = -v_t * scale_factor  # [batch_size, 3]

    # Rotate tangent to body frame
    tb = rotate_vectors_inverse_batch(t, q)  # [batch_size, 3]

    # Compute denominator for impulse calculation
    r_cross_tb = torch.cross(r, tb, dim=1)  # [batch_size, 3]
    I_inv_cross = torch.bmm(I_inv, r_cross_tb.unsqueeze(2)).squeeze(
        2
    )  # [batch_size, 3]
    r_cross_I_inv_cross = torch.cross(I_inv_cross, r, dim=1)  # [batch_size, 3]
    dot_products = torch.sum(
        r_cross_I_inv_cross * tb, dim=1, keepdim=True
    )  # [batch_size, 1]
    denominator = m_inv + dot_products  # [batch_size, 1]

    # Compute impulses
    J_friction = delta_v_friction / denominator  # [batch_size, 3]

    # Compute velocity changes
    dv = J_friction * m_inv  # [batch_size, 3]
    J_body = rotate_vectors_inverse_batch(J_friction, q)  # [batch_size, 3]
    r_cross_J = torch.cross(r, J_body, dim=1)  # [batch_size, 3]
    dw = torch.bmm(I_inv, r_cross_J.unsqueeze(2)).squeeze(2)  # [batch_size, 3]

    # Combine angular and linear velocity changes
    dbody_qd = torch.cat([dw, dv], dim=1)  # [batch_size, 6]

    # Only apply corrections to contacts with sufficient tangential velocity
    dbody_qd = dbody_qd * mask.repeat(1, 6)

    return dbody_qd
