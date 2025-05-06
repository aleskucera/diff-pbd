import time
from typing import Tuple

from demos.utils import *
from pbd_torch.constants import *
from pbd_torch.transform import *


def positional_deltas(
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
    dot_a = torch.sum(final_cross_a * n_a, dim=1, keepdim=True)  # [B, 1, 1]
    dot_b = torch.sum(final_cross_b * n_b, dim=1, keepdim=True)  # [B, 1, 1]

    # Compute weights
    weight_a = m_a_inv + dot_a  # [B, 1, 1]
    weight_b = m_b_inv + dot_b  # [B, 1, 1]

    # Compute the impulse magnitude
    d_lambda = -c / (weight_a + weight_b) # [B, 1, 1]

    # Compute impulse vectors
    p = d_lambda * n  # [B, 3, 1]
    p_a = d_lambda * n_a  # [B, 3, 1]
    p_b = d_lambda * n_b  # [B, 3, 1]

    # Positional corrections
    dx_a = p * m_a_inv  # [B, 3, 1]
    dx_b = -p * m_b_inv  # [B, 3, 1]

    # Rotational corrections
    r_cross_p_a = torch.cross(r_a, p_a, dim=1)  # [B, 3, 1]
    r_cross_p_b = torch.cross(r_b, p_b, dim=1)  # [B, 3, 1]

    w_a = rotate_vectors_batch(torch.bmm(I_a_inv, r_cross_p_a), q_a)  # [B, 3, 1]
    w_b = rotate_vectors_batch(torch.bmm(I_b_inv, r_cross_p_b), q_b)  # [B, 3, 1]

    # Create quaternion vectors for rotational corrections
    zeros = torch.zeros((batch_size, 1, 1), device=device)
    w_a_quat = torch.cat([zeros, w_a], dim=1)  # [B, 4, 1]
    w_b_quat = torch.cat([zeros, w_b], dim=1)  # [B, 4, 1]

    # Quaternion multiplication
    dq_a = 0.5 * quat_mul_batch(w_a_quat, q_a)  # [B, 4, 1]
    dq_b = -0.5 * quat_mul_batch(w_b_quat, q_b)  # [B, 4, 1]

    # Combine positional and rotational corrections
    dbody_trans_a = torch.cat([dx_a, dq_a], dim=1)  # [B, 7, 1]
    dbody_trans_b = torch.cat([dx_b, dq_b], dim=1)  # [B, 7, 1]

    return dbody_trans_a, dbody_trans_b, d_lambda


def joint_angular_deltas(
    body_q_p: torch.Tensor,  # [B, 7, 1]
    body_q_c: torch.Tensor,  # [B, 7, 1]
    X_p: torch.Tensor,  # [B, 7, 1]
    X_c: torch.Tensor,  # [B, 7, 1]
    joint_axis: torch.Tensor,  # [B, 3, 1]
    I_p_inv: torch.Tensor,  # [B, 3, 3]
    I_c_inv: torch.Tensor,  # [B, 3, 3]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched version of joint_angular_delta that processes multiple joints at once."""
    batch_size = body_q_p.shape[0]
    device = body_q_p.device

    # Extract orientations
    q_p = body_q_p[:, 3:]  # [B, 4, 1]
    q_c = body_q_c[:, 3:]  # [B, 4, 1]

    # Transform joint frames to world space
    X_wj_p = transform_multiply_batch(body_q_p, X_p)  # [B, 7, 1]
    X_wj_c = transform_multiply_batch(body_q_c, X_c)  # [B, 7, 1]

    # Rotate joint axes to world space
    axis_w_p = rotate_vectors_batch(joint_axis, X_wj_p[:, 3:])  # [B, 3, 1]
    axis_w_c = rotate_vectors_batch(joint_axis, X_wj_c[:, 3:])  # [B, 3, 1]

    # Compute rotation vector and angle
    rot_vector = torch.cross(axis_w_p, axis_w_c, dim=1)  # [B, 3, 1]
    theta = torch.norm(rot_vector, dim=1, keepdim=True)  # [B, 1, 1]

    # Initialize outputs
    dq_p = torch.zeros((batch_size, 4, 1), device=device) # [B, 4, 1]
    dq_c = torch.zeros((batch_size, 4, 1), device=device) # [B, 4, 1]

    # Create mask for non-zero thetas to avoid division by zero
    mask = (theta > 1e-6).flatten() # [B]

    # Process only the joints with significant angles
    n = torch.zeros_like(rot_vector) # [B, 3, 1]
    n[mask] = rot_vector[mask] / theta[mask]  # [B, 3, 1]

    # Rotate normals to body frames
    n_p = rotate_vectors_inverse_batch(n, q_p)  # [B, 3, 1]
    n_c = rotate_vectors_inverse_batch(n, q_c)  # [B, 3, 1]

    # Compute weights using matrix-vector multiplications
    # I_inv @ n
    I_p_inv_n = torch.bmm(I_p_inv, n_p)  # [B, 3, 1]
    I_c_inv_n = torch.bmm(I_c_inv, n_c)  # [B, 3, 1]

    # Dot products for weights: n_p @ I_p_inv @ n_p and n_c @ I_c_inv @ n_c
    weight_p = torch.sum(n_p * I_p_inv_n, dim=1, keepdim=True)  # [B, 1, 1]
    weight_c = torch.sum(n_c * I_c_inv_n, dim=1, keepdim=True)  # [B, 1, 1]

    weight = weight_p + weight_c  # [B, 1, 1]

    # Compute angle distributions
    theta_p = torch.zeros_like(theta) # [B, 1, 1]
    theta_c = torch.zeros_like(theta) # [B, 1, 1]

    valid_weights = mask & (weight.flatten() > 1e-6) # [B]
    theta_p[valid_weights] = (
        theta[valid_weights] * weight_p[valid_weights] / weight[valid_weights]
    ) # [B, 1, 1]
    theta_c[valid_weights] = (
        -theta[valid_weights] * weight_c[valid_weights] / weight[valid_weights]
    ) # [B, 1, 1]

    # Create quaternion vectors for rotational corrections
    zeros = torch.zeros((batch_size, 1, 1), device=device) # [B, 1, 1]

    # Multiply n by theta_p and theta_c
    n_p_scaled = n * theta_p  # [B, 3, 1]
    n_c_scaled = n * theta_c  # [B, 3, 1]

    w_p_quat = torch.cat([zeros, n_p_scaled], dim=1)  # [B, 4, 1]
    w_c_quat = torch.cat([zeros, n_c_scaled], dim=1)  # [B, 4, 1]

    # Quaternion multiplication
    dq_p = 0.5 * quat_mul_batch(w_p_quat, q_p)  # [B, 4, 1]
    dq_c = 0.5 * quat_mul_batch(w_c_quat, q_c)  # [B, 4, 1]

    return dq_p, dq_c


def joint_deltas(
    body_q_p: torch.Tensor,  # [B, 7, 1]
    body_q_c: torch.Tensor,  # [B, 7, 1]
    X_p: torch.Tensor,  # [B, 7, 1]
    X_c: torch.Tensor,  # [B, 7, 1]
    joint_axis: torch.Tensor,  # [B, 3, 1]
    m_p_inv: torch.Tensor,  # [B, 1, 1]
    m_c_inv: torch.Tensor,  # [B, 1, 1]
    I_p_inv: torch.Tensor,  # [B, 3, 3]
    I_c_inv: torch.Tensor,  # [B, 3, 3]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched version of joint_delta that processes multiple joints at once."""

    # Get angular corrections
    dq_p, dq_c = joint_angular_deltas(
        body_q_p, body_q_c, X_p, X_c, joint_axis, I_p_inv, I_c_inv
    )

    # Create copies of body positions/orientations
    new_body_q_p = body_q_p.clone()
    new_body_q_c = body_q_c.clone()

    # Apply angular corrections
    new_body_q_p[:, 3:] = normalize_quat_batch(new_body_q_p[:, 3:] + dq_p)
    new_body_q_c[:, 3:] = normalize_quat_batch(new_body_q_c[:, 3:] + dq_c)

    # Get positional corrections
    dbody_q_p, dbody_q_c, _ = positional_deltas(
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


def ground_restitution_deltas(
    body_q: torch.Tensor,  # [C, 7, 1]
    body_qd: torch.Tensor,  # [C, 6, 1]
    body_qd_prev: torch.Tensor,  # [C, 6, 1]
    r: torch.Tensor,  # [C, 3, 1]
    n: torch.Tensor,  # [C, 3, 1]
    m_inv: torch.Tensor,  # [C, 1]
    I_inv: torch.Tensor,  # [C, 3, 3]
    restitution: torch.Tensor,  # [C, 1]
):
    """Batched version of ground_restitution_delta that processes multiple contacts at once."""
    batch_size = body_q.shape[0]
    device = body_q.device

    # Initialize outputs
    dbody_qd = torch.zeros((batch_size, 6, 1), device=device) # [C, 6]

    # Extract orientations and velocities
    q = body_q[:, 3:]  # Orientation [C, 4, 1]
    v = body_qd[:, 3:]  # Linear velocity [C, 3, 1]
    w = body_qd[:, :3]  # Angular velocity [C, 3, 1]
    v_prev = body_qd_prev[:, 3:]  # Previous linear velocity [C, 3, 1]
    w_prev = body_qd_prev[:, :3]  # Previous angular velocity [C, 3, 1]

    # Compute cross products of angular velocity and contact point
    w_cross_r = torch.cross(w, rotate_vectors_batch(r, q), dim=1)  # [C, 3, 1]
    w_prev_cross_r = torch.cross(w_prev, rotate_vectors_batch(r, q), dim=1)  # [C, 3, 1]

    # Compute the relative velocities
    v_rel = v + w_cross_r  # [C, 3, 1]
    v_rel_prev = v_prev + w_prev_cross_r  # [C, 3, 1]

    # Compute normal components of relative velocities
    vn = torch.sum(v_rel * n, dim=1, keepdim=True)  # [C, 1, 1]
    vn_prev = torch.sum(v_rel_prev * n, dim=1, keepdim=True)  # [C, 1, 1]

    # Ensure restitution has correct shape
    if restitution.dim() == 1:
        restitution = restitution.unsqueeze(1)  # [C, 1, 1]

    # Compute velocity corrections
    zeros = torch.zeros_like(vn) # [C, 1, 1]
    velocity_correction = torch.max(
        zeros, vn + restitution.unsqueeze(1) * vn_prev
    )  # [C, 1, 1]

    # Apply threshold to avoid jittering (2 * gravity * dt)
    jitter_threshold = torch.tensor(2 * 9.81 * 0.01, device=device)
    mask = (torch.abs(velocity_correction) >= jitter_threshold).flatten()  # [C]

    if not mask.any():
        return dbody_qd

    # Compute the impulse direction (negative normal)
    delta_v_restitution = -n * velocity_correction  # [C, 3, 1]

    # Rotate normals to body frames
    nb = rotate_vectors_inverse_batch(n, q)  # [C, 3, 1]

    # Compute denominator for impulse calculation
    r_cross_nb = torch.cross(r, nb, dim=1)  # [C, 3, 1]
    I_inv_cross = torch.bmm(I_inv, r_cross_nb) # [C, 3, 1]
    r_cross_I_inv_cross = torch.cross(I_inv_cross, r, dim=1)  # [C, 3, 1]
    dot_products = torch.sum(r_cross_I_inv_cross * nb, dim=1, keepdim=True)  # [C, 1, 1]
    denominator = m_inv + dot_products  # [C, 1, 1]

    # Compute impulses
    J_restitution = delta_v_restitution / denominator  # [C, 3, 1]

    # Compute velocity changes
    dv = J_restitution * m_inv  # [C, 3, 1]
    J_body = rotate_vectors_inverse_batch(J_restitution, q)  # [C, 3, 1]
    r_cross_J = torch.cross(r, J_body, dim=1)  # [C, 3, 1]
    dw = rotate_vectors_batch(torch.bmm(I_inv, r_cross_J), q)  # [C, 3, 1]

    # Combine angular and linear velocity changes
    dbody_qd = torch.cat([dw, dv], dim=1)  # [C, 6, 1]

    # Only apply corrections to contacts that exceeded the jitter threshold
    dbody_qd = dbody_qd * mask.view(-1, 1, 1)  # [C, 6, 1]

    return dbody_qd


def ground_dynamic_friction_deltas(
    body_q: torch.Tensor,  # [B, 7, 1]
    body_qd: torch.Tensor,  # [B, 6, 1]
    r: torch.Tensor,  # [B, 3, 1]
    n: torch.Tensor,  # [B, 3, 1]
    m_inv: torch.Tensor,  # [B, 1, 1]
    I_inv: torch.Tensor,  # [B, 3, 3]
    dynamic_friction: torch.Tensor,  # [B, 1]
    lambda_n: torch.Tensor,  # [B, 1]
    dt: float,
):
    """Batched version of ground_dynamic_friction_delta that processes multiple contacts at once."""
    batch_size = body_q.shape[0]
    device = body_q.device

    # Initialize outputs
    dbody_qd = torch.zeros((batch_size, 6, 1), device=device)

    # Extract orientations and velocities
    q = body_q[:, 3:]  # Orientation [batch_size, 4, 1]
    v = body_qd[:, 3:]  # Linear velocity [batch_size, 3, 1]
    w = body_qd[:, :3]  # Angular velocity [batch_size, 3, 1]

    # Compute cross products of angular velocity and contact point
    w_cross_r = torch.cross(w, rotate_vectors_batch(r, q), dim=1)  # [batch_size, 3, 1]

    # Compute the relative velocities
    v_rel = v + w_cross_r  # [batch_size, 3, 1]

    # Compute normal component and project to get tangential component
    vn_components = torch.sum(v_rel * n, dim=1, keepdim=True) * n  # [batch_size, 3, 1]
    v_t = v_rel - vn_components  # [batch_size, 3, 1]

    # Compute tangential velocity magnitude
    v_t_magnitude = torch.norm(v_t, dim=1, keepdim=True)  # [batch_size, 1, 1]

    # Create mask for non-zero tangential velocities
    mask = (v_t_magnitude >= 1e-5).flatten() # [batch_size]

    if not mask.any():
        return dbody_qd

    # Compute normalized tangent direction
    t = torch.zeros_like(v_t) # [batch_size, 3, 1]
    # t[mask.repeat(1, 3)] = v_t[mask.repeat(1, 3)] / v_t_magnitude[mask].repeat(1, 3)
    t[mask] = v_t[mask] / v_t_magnitude[mask] # [batch_size, 3, 1]

    # Compute Coulomb friction
    coulomb_friction = torch.abs(
        dynamic_friction * lambda_n / (dt**2)
    ).unsqueeze(1)  # [batch_size, 1, 1]

    # Compute velocity change due to friction
    min_values = torch.min(coulomb_friction, v_t_magnitude)  # [batch_size, 1, 1]
    scale_factor = min_values / v_t_magnitude  # [batch_size, 1, 1]
    delta_v_friction = -v_t * scale_factor  # [batch_size, 3, 1]

    # Rotate tangent to body frame
    tb = rotate_vectors_inverse_batch(t, q)  # [batch_size, 3, 1]

    # Compute denominator for impulse calculation
    r_cross_tb = torch.cross(r, tb, dim=1)  # [batch_size, 3, 1]
    I_inv_cross = torch.bmm(I_inv, r_cross_tb) # [batch_size, 3, 1]
    r_cross_I_inv_cross = torch.cross(I_inv_cross, r, dim=1)  # [batch_size, 3, 1]
    dot_products = torch.sum(r_cross_I_inv_cross * tb, dim=1, keepdim=True)  # [batch_size, 1, 1]
    denominator = m_inv + dot_products  # [batch_size, 1, 1]

    # Compute impulses
    J_friction = delta_v_friction / denominator  # [batch_size, 3, 1]

    # Compute velocity changes
    dv = J_friction * m_inv  # [batch_size, 3, 1]
    J_body = rotate_vectors_inverse_batch(J_friction, q)  # [batch_size, 3, 1]
    r_cross_J = torch.cross(r, J_body, dim=1)  # [batch_size, 3, 1]
    dw = rotate_vectors_batch(torch.bmm(I_inv, r_cross_J), q)  # [batch_size, 3, 1]

    # Combine angular and linear velocity changes
    dbody_qd = torch.cat([dw, dv], dim=1)  # [batch_size, 6, 1]

    # Only apply corrections to contacts with sufficient tangential velocity
    dbody_qd = dbody_qd * mask.view(-1, 1, 1)  # [batch_size, 6, 1]

    return dbody_qd

def velocity_deltas(
    body_q: torch.Tensor,  # [batch_size, 7, 1]
    body_qd: torch.Tensor,  # [batch_size, 6, 1]
    body_qd_prev: torch.Tensor,  # [batch_size, 6, 1]
    r: torch.Tensor,  # [batch_size, 3, 1]
    n: torch.Tensor,  # [batch_size, 3, 1]
    m_inv: torch.Tensor,  # [batch_size, 1, 1]
    I_inv: torch.Tensor,  # [batch_size, 3, 3]
    restitution: torch.Tensor,  # [batch_size, 1]
    dynamic_friction: torch.Tensor,  # [batch_size, 1]
    lambda_n: torch.Tensor,  # [batch_size, 1]
    dt: float,
):
    device = body_q.device

    # Compute the contact point velocity in the world space
    v_n, v_t_direction, v_t_magnitude = body_point_velocity(body_q, body_qd, r, n)
    v_n_prev, _, _ = body_point_velocity(body_q, body_qd_prev, r, n)

    # ------------------------------ START: RESTITUTION ------------------------------
    c_restitution = torch.min(- restitution.view(-1, 1, 1) * v_n_prev, torch.zeros_like(v_n))  # [batch_size, 1, 1]

    # Apply a threshold to avoid jittering (2 * gravity * dt)
    jitter_threshold = torch.tensor(2 * 9.81 * dt, device=device)
    mask = (torch.abs(c_restitution) < jitter_threshold).flatten()  # [batch_size]
    c_restitution[mask] = 0.0  # Set non-masked values to zero

    dbody_qd_rest = velocity_deltas_from_constraint(body_q, c_restitution, n, m_inv, I_inv, r)  # [batch_size, 6, 1]
    # ------------------------------- END: RESTITUTION ------------------------------

    # ------------------------------ START: DYNAMIC FRICTION ------------------------------
    c_friction = torch.min((dynamic_friction * lambda_n).view(-1, 1, 1) / dt, v_t_magnitude)  # [batch_size, 1, 1]

    dbody_qd_friction = velocity_deltas_from_constraint(body_q, c_friction, v_t_direction, m_inv, I_inv, r) # [batch_size, 6, 1]
    # ------------------------------------ END: DYNAMIC FRICTION ------------------------------

    # Combine restitution and friction deltas
    dbody_qd = dbody_qd_rest + dbody_qd_friction # [batch_size, 6, 1]

    return dbody_qd

def velocity_deltas_from_constraint(
    body_q: torch.Tensor,  # [batch_size, 7, 1]
    c: torch.Tensor,  # [batch_size, 1, 1]
    J: torch.Tensor,  # [batch_size, 3, 1]
    m_inv: torch.Tensor,  # [batch_size, 1, 1]
    I_inv: torch.Tensor,  # [batch_size, 3, 3]
    r: torch.Tensor,  # [batch_size, 3, 1]
):
    gen_inverse_mass = generalized_inverse_mass(body_q, m_inv, I_inv, r, J)  # [batch_size, 1, 1]

    p = (-c / gen_inverse_mass) * J  # [batch_size, 3, 1]
    p_b = rotate_vectors_inverse_batch(p, body_q[:, 3:])  # [batch_size, 3, 1]

    # Linear velocity changes
    dv = p * m_inv  # [batch_size, 3, 1]

    r_cross_p = torch.cross(r, p_b, dim=1)  # [batch_size, 3, 1]
    dw_b = torch.matmul(I_inv, r_cross_p) # [batch_size, 3, 1]
    dw = rotate_vectors_batch(dw_b, body_q[:, 3:])  # [batch_size, 3, 1]

    dbody_qd = torch.cat([dw, dv], dim=1)  # [batch_size, 6, 1]

    return dbody_qd


def body_point_velocity(
        body_q: torch.Tensor,  # [batch_size, 7, 1]
        body_qd: torch.Tensor,  # [batch_size, 6, 1]
        r: torch.Tensor,  # [batch_size, 3, 1]
        n: torch.Tensor,  # [batch_size, 3, 1]
):
    q = body_q[:, 3:]  # Orientation [batch_size, 4, 1]
    v = body_qd[:, 3:]  # Linear velocity [batch_size, 3, 1]
    w = body_qd[:, :3]  # Angular velocity [batch_size, 3, 1]

    w_cross_r = torch.cross(w, rotate_vectors_batch(r, q), dim=1)  # [batch_size, 3, 1]

    v_point = v + w_cross_r  # [batch_size, 3, 1]

    # Compute normal component and project to get tangential component
    v_n = torch.matmul(v.transpose(2, 1), v_point)  # [batch_size, 1, 1]

    v_t = v_point - v_n * n  # [batch_size, 3, 1]
    v_t_magnitude = torch.norm(v_t, dim=1, keepdim=True)  # [batch_size, 1, 1]
    mask =  (v_t_magnitude >= 1e-5).flatten()  # [batch_size]
    v_t_direction = torch.zeros_like(v_t)  # [batch_size, 3, 1]
    v_t_direction[mask] = v_t[mask] / v_t_magnitude[mask]  # [batch_size, 3, 1]

    return v_n, v_t_direction, v_t_magnitude

def generalized_inverse_mass(
        body_q: torch.Tensor,  # [batch_size, 7, 1]
        m_inv: torch.Tensor,  # [batch_size, 1, 1]
        I_inv: torch.Tensor,  # [batch_size, 3, 3]
        r: torch.Tensor,  # [batch_size, 3, 1]
        J: torch.Tensor,  # [batch_size, 3, 1]
):
    q = body_q[:, 3:]  # Orientation [batch_size, 4, 1]

    n_b = rotate_vectors_inverse_batch(J, q)  # [batch_size, 3, 1]

    r_cross_n_b = torch.cross(r, n_b, dim=1)  # [batch_size, 3, 1]
    I_inv_cross = torch.matmul(I_inv, r_cross_n_b)  # [batch_size, 3, 1]
    angular_w = torch.matmul(r_cross_n_b.transpose(2, 1), I_inv_cross)  # [batch_size, 1, 1]
    return m_inv + angular_w # [batch_size, 1, 1]





