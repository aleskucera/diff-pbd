import time
from typing import Tuple

from demos.utils import *
from pbd_torch.constants import *
from pbd_torch.transform import *


def generalized_inverse_mass(
        body_q: torch.Tensor,  # [batch_size, 7, 1]
        m_inv: torch.Tensor,  # [batch_size, 1, 1]
        I_inv: torch.Tensor,  # [batch_size, 3, 3]
        r: torch.Tensor,  # [batch_size, 3, 1]
        J: torch.Tensor,  # [batch_size, 3, 1]
):
    q = body_q[:, 3:]  # Orientation [batch_size, 4, 1]

    J_b = rotate_vectors_inverse_batch(J, q)  # [batch_size, 3, 1]

    r_cross_J = torch.cross(r, J_b, dim=1)  # [batch_size, 3, 1]
    I_inv_cross = torch.matmul(I_inv, r_cross_J)  # [batch_size, 3, 1]
    # angular_component = torch.matmul(r_cross_J.transpose(2, 1), I_inv_cross)  # [batch_size, 1, 1]

    angular_component = torch.sum(torch.cross(I_inv_cross, r, dim=1) * J_b, dim=1, keepdim=True)  # [batch_size, 1, 1]

    return m_inv + angular_component # [batch_size, 1, 1]

def generalized_inverse_angular_mass(
        body_q: torch.Tensor, # [batch_size, 7, 1]
        I_inv: torch.Tensor,  # [batch_size, 3, 3]
        J: torch.Tensor,  # [batch_size, 3, 1]
):
    J_b = rotate_vectors_inverse_batch(J, body_q[:, 3:])  # [batch_size, 3, 1]
    J_b_T = J_b.transpose(2, 1)  # [batch_size, 1, 3]
    return torch.matmul(J_b_T, torch.matmul(I_inv, J_b))  # [batch_size, 3, 1]

def positional_constraint(
    body_trans_a: torch.Tensor,  # [batch_size, 7, 1]
    body_trans_b: torch.Tensor,  # [batch_size, 7, 1]
    r_a: torch.Tensor,  # [batch_size, 3, 1]
    r_b: torch.Tensor,  # [batch_size, 3, 1]
):
    r_a_world = transform_points_batch(r_a, body_trans_a)
    r_b_world = transform_points_batch(r_b, body_trans_b)

    delta_x = r_a_world - r_b_world  # [batch_size, 3, 1]

    c = torch.norm(delta_x, dim=1, keepdim=True)  # [batch_size, 1, 1]
    J = delta_x / (c + 1e-6)  # [batch_size, 3, 1]

    return c, J

def positional_deltas_from_impulse(
    p: torch.Tensor, # [batch_size, 3, 1]
    body_trans: torch.Tensor, # [batch_size, 7, 1]
    r: torch.Tensor, # [batch_size, 3, 1]
    m_inv: torch.Tensor, # [batch_size, 1, 1]
    I_inv: torch.Tensor # [batch_size, 3, 3]
):
    device = p.device
    batch_size = p.shape[0]

    q = body_trans[:, 3:] # [batch_size, 4, 1]

    # Get impulse in body frame
    p_b = rotate_vectors_inverse_batch(p, q) # [batch_size, 3, 1]

    # Positional correction
    dx = p * m_inv # [batch_size, 3, 1]

    # Rotational correction
    r_cross_p = torch.cross(r, p_b, dim=1) # [batch_size, 3, 1]
    omega = rotate_vectors_batch(torch.matmul(I_inv, r_cross_p), q) # [batch_size, 3, 1]
    zeros = torch.zeros((batch_size, 1, 1), device=device) # [batch_size, 1, 1]
    omega_quat = torch.cat([zeros, omega], dim=1) # [batch_size, 4, 1]
    dq = 0.5 * quat_mul_batch(omega_quat, q) # [batch_size, 4, 1]

    # Concatenate positional and rotational corrections
    dbody_trans = torch.cat([dx, dq], dim=1) # [batch_size, 7, 1]

    return dbody_trans

def positional_deltas(
    body_trans_a: torch.Tensor,  # [batch_size, 7, 1]
    body_trans_b: torch.Tensor,  # [batch_size, 7, 1]
    r_a: torch.Tensor,  # [batch_size, 3, 1]
    r_b: torch.Tensor,  # [batch_size, 3, 1]
    m_a_inv: torch.Tensor,  # [batch_size, 1, 1]
    m_b_inv: torch.Tensor,  # [batch_size, 1, 1]
    I_a_inv: torch.Tensor,  # [batch_size, 3, 3]
    I_b_inv: torch.Tensor,  # [batch_size, 3, 3]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c, J = positional_constraint(body_trans_a, body_trans_b, r_a, r_b) # [batch_size, 1, 1], [batch_size, 3, 1]

    weight_a = generalized_inverse_mass(body_trans_a, m_a_inv, I_a_inv, r_a, J)
    weight_b = generalized_inverse_mass(body_trans_b, m_b_inv, I_b_inv, r_b, J)

    # Compute the impulse magnitude
    d_lambda = -c / (weight_a + weight_b) # [batch_size, 1, 1]

    # Compute impulse vectors
    p = d_lambda * J  # [batch_size, 3, 1]

    # Combine positional and rotational corrections
    dbody_trans_a = positional_deltas_from_impulse(p, body_trans_a, r_a, m_a_inv, I_a_inv) # [batch_size, 7, 1]
    dbody_trans_b = positional_deltas_from_impulse(-p, body_trans_b, r_b, m_b_inv, I_b_inv) # [batch_size, 7, 1]

    return dbody_trans_a, dbody_trans_b, d_lambda

def joint_constraint(
    body_q_p: torch.Tensor,  # [batch_size, 7, 1]
    body_q_c: torch.Tensor,  # [batch_size, 7, 1]
    X_p: torch.Tensor,  # [batch_size, 7, 1]
    X_c: torch.Tensor,  # [batch_size, 7, 1]
    joint_axis: torch.Tensor,  # [batch_size, 3, 1]
):
    X_wj_p = transform_multiply_batch(body_q_p, X_p)  # [batch_size, 7, 1]
    X_wj_c = transform_multiply_batch(body_q_c, X_c)  # [batch_size, 7, 1]

    axis_w_p = rotate_vectors_batch(joint_axis, X_wj_p[:, 3:])  # [batch_size, 3, 1]
    axis_w_c = rotate_vectors_batch(joint_axis, X_wj_c[:, 3:])  # [batch_size, 3, 1]

    rot_vector = torch.cross(axis_w_c, axis_w_p, dim=1)  # [batch_size, 3, 1]
    c = torch.norm(rot_vector, dim=1, keepdim=True) # [batch_size, 1, 1]
    J = rot_vector / (c + 1e-6) # [batch_size, 3, 1]

    return c, J

def joint_angular_deltas_from_impulse(
    p: torch.Tensor,  # [batch_size, 3, 1]
    body_q: torch.Tensor,  # [batch_size, 7, 1]
    I_inv: torch.Tensor,  # [batch_size, 3, 3]
):
    device = p.device
    batch_size = p.shape[0]

    q = body_q[:, 3:]  # [batch_size, 4, 1]
    p_b = rotate_vectors_inverse_batch(p, q)  # [batch_size, 3, 1]

    # Rotational correction
    zeros = torch.zeros((batch_size, 1, 1), device=device)  # [batch_size, 1, 1]
    omega = rotate_vectors_batch(torch.matmul(I_inv, p_b), q)  # [batch_size, 3, 1]
    omega_quat = torch.cat([zeros, omega], dim=1)  # [batch_size, 4, 1]
    dq = 0.5 * quat_mul_batch(omega_quat, q)  # [batch_size, 4, 1]

    return dq

def joint_angular_deltas(
    body_q_p: torch.Tensor,  # [batch_size, 7, 1]
    body_q_c: torch.Tensor,  # [batch_size, 7, 1]
    X_p: torch.Tensor,  # [batch_size, 7, 1]
    X_c: torch.Tensor,  # [batch_size, 7, 1]
    joint_axis: torch.Tensor,  # [batch_size, 3, 1]
    I_p_inv: torch.Tensor,  # [batch_size, 3, 3]
    I_c_inv: torch.Tensor,  # [batch_size, 3, 3]
) -> Tuple[torch.Tensor, torch.Tensor]:
    c, J = joint_constraint(
        body_q_p, body_q_c, X_p, X_c, joint_axis
    )  # [batch_size, 1, 1], [batch_size, 3, 1]

    weight_p = generalized_inverse_angular_mass(body_q_p, I_p_inv, J)  # [batch_size, 1, 1]
    weight_c = generalized_inverse_angular_mass(body_q_c, I_c_inv, J)  # [batch_size, 1, 1]
    weight = weight_p + weight_c  # [batch_size, 1, 1]

    dlambda = -c / weight # [batch_size, 1, 1]
    p = dlambda * J # [batch_size, 3, 1]

    dq_p = joint_angular_deltas_from_impulse(p, body_q_p, I_p_inv)  # [batch_size, 4, 1]
    dq_c = joint_angular_deltas_from_impulse(-p, body_q_c, I_c_inv)  # [batch_size, 4, 1]

    return dq_p, dq_c

def joint_deltas(
    body_q_p: torch.Tensor,  # [batch_size, 7, 1]
    body_q_c: torch.Tensor,  # [batch_size, 7, 1]
    X_p: torch.Tensor,  # [batch_size, 7, 1]
    X_c: torch.Tensor,  # [batch_size, 7, 1]
    joint_axis: torch.Tensor,  # [batch_size, 3, 1]
    m_p_inv: torch.Tensor,  # [batch_size, 1, 1]
    m_c_inv: torch.Tensor,  # [batch_size, 1, 1]
    I_p_inv: torch.Tensor,  # [batch_size, 3, 3]
    I_c_inv: torch.Tensor,  # [batch_size, 3, 3]
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Get angular corrections
    dq_p, dq_c = joint_angular_deltas(
        body_q_p, body_q_c, X_p, X_c, joint_axis, I_p_inv, I_c_inv
    )

    # Apply angular corrections
    body_q_p[:, 3:] = normalize_quat_batch(body_q_p[:, 3:] + dq_p)
    body_q_c[:, 3:] = normalize_quat_batch(body_q_c[:, 3:] + dq_c)

    # Get positional corrections
    dbody_q_p, dbody_q_c, _ = positional_deltas(
        body_trans_a=body_q_p,
        body_trans_b=body_q_c,
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

def velocity_deltas_from_constraint(
        c: torch.Tensor,  # [batch_size, 1, 1]
        J: torch.Tensor,  # [batch_size, 3, 1]
        body_q: torch.Tensor,  # [batch_size, 7, 1]
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
    v_t_direction = v_t / (v_t_magnitude + 1e-6)  # [batch_size, 3, 1]

    return v_n, v_t_direction, v_t_magnitude

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

    dbody_qd_rest = velocity_deltas_from_constraint(c_restitution, n, body_q, m_inv, I_inv, r)  # [batch_size, 6, 1]
    # ------------------------------- END: RESTITUTION ------------------------------

    # ------------------------------ START: DYNAMIC FRICTION ------------------------------
    c_friction = torch.min((dynamic_friction * lambda_n).view(-1, 1, 1) / dt, v_t_magnitude)  # [batch_size, 1, 1]

    dbody_qd_friction = velocity_deltas_from_constraint(c_friction, v_t_direction, body_q, m_inv, I_inv, r) # [batch_size, 6, 1]
    # ------------------------------------ END: DYNAMIC FRICTION ------------------------------

    # Combine restitution and friction deltas
    dbody_qd = dbody_qd_rest + dbody_qd_friction # [batch_size, 6, 1]

    return dbody_qd






