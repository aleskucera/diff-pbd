from typing import Tuple

from pbd_torch.transform import *
from demos.utils import *

def positional_delta(
        body_q_a: torch.Tensor, body_q_b: torch.Tensor, r_a: torch.Tensor,
        r_b: torch.Tensor, m_a_inv: torch.Tensor, m_b_inv: torch.Tensor,
        I_a_inv: torch.Tensor, I_b_inv: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dbody_q_a = torch.zeros(7, device=body_q_a.device)  # Linear and angular correction for body A
    dbody_q_b = torch.zeros(7, device=body_q_a.device)  # Linear and angular correction for body B
    d_lambda = torch.tensor([0.0], device=body_q_a.device)  # Impulse magnitude

    q_a = body_q_a[3:]  # Orientation of body A
    q_b = body_q_b[3:]  # Orientation of body B

    # Compute the relative position of the contact point in the world frame
    delta_x = transform(r_a, body_q_a) - transform(r_b, body_q_b)

    c = torch.norm(delta_x)
    if c < 1e-6:
        return dbody_q_a, dbody_q_b, d_lambda

    n = delta_x / c  # Normal vector pointing from body B to body A in the world frame
    n_a = rotate_vectors_inverse(n,
                                 q_a)  # Rotate the normal to the body A frame
    n_b = rotate_vectors_inverse(n,
                                 q_b)  # Rotate the normal to the body B frame

    weight_a = m_a_inv + torch.dot(
        torch.linalg.cross(I_a_inv @ torch.linalg.cross(r_a, n_a), r_a), n_a)
    weight_b = m_b_inv + torch.dot(
        torch.linalg.cross(I_b_inv @ torch.linalg.cross(r_b, n_b), r_b), n_b)

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
        r_a, p_a)  # Angular velocity correction for body A
    w_a_quat = torch.cat([torch.tensor([0.0], device=w_a.device), w_a], dim=0)
    dq_a = 0.5 * quat_mul(q_a, w_a_quat)

    w_b = I_b_inv @ torch.linalg.cross(
        r_b, p_b)  # Angular velocity correction for body B
    w_b_quat = torch.cat([torch.tensor([0.0], device=w_b.device), w_b], dim=0)
    dq_b = -0.5 * quat_mul(q_b, w_b_quat)

    dbody_q_a = torch.cat([dx_a, dq_a])
    dbody_q_b = torch.cat([dx_b, dq_b])

    return dbody_q_a, dbody_q_b, d_lambda


def ground_restitution_delta(body_q: torch.Tensor, body_qd: torch.Tensor,
                             body_qd_prev: torch.Tensor, r: torch.Tensor,
                             n: torch.Tensor, m_inv: torch.Tensor,
                             I_inv: torch.Tensor, restitution: torch.Tensor):
    dbody_qd = torch.zeros(6, device=body_q.device)  # Linear and angular correction

    q = body_q[3:]  # Orientation
    v = body_qd[3:]  # Linear velocity
    w = body_qd[:3]  # Angular velocity
    v_prev = body_qd_prev[3:]  # Previous linear velocity
    w_prev = body_qd_prev[:3]  # Previous angular velocity

    # Compute the normal component of the relative velocity
    v_rel = v + rotate_vectors(torch.linalg.cross(w, r), q)
    vn = torch.dot(v_rel, n)

    # Compute the normal component of the relative velocity before the velocity update
    v_rel_prev = v_prev + rotate_vectors(torch.linalg.cross(w_prev, r), q)
    vn_prev = torch.dot(v_rel_prev, n)
    # vn_prev = torch.max(vn_prev, torch.tensor([0.0]))

    velocity_correction = torch.max(torch.tensor([0.0], device=vn.device),
                                    (vn + restitution * vn_prev))

    # If the correction is too small we can ignore it to avoid jittering
    if torch.abs(velocity_correction) < 2 * 9.81 * 0.01:  # 2 * gravity * dt
        return dbody_qd

    # Compute the change of velocity due to the restitution
    delta_v_restitution = -n * velocity_correction

    # Compute the impulse due to the restitution in the world frame
    nb = rotate_vectors_inverse(n, q)  # Rotate the normal to the body frame
    J_restitution = delta_v_restitution / (m_inv + torch.dot(
        torch.linalg.cross(I_inv @ torch.linalg.cross(r, nb), r), nb))
    dv = J_restitution * m_inv
    dw = I_inv @ torch.linalg.cross(r, rotate_vectors_inverse(
        J_restitution, q))

    dbody_qd = torch.cat([dw, dv])
    return dbody_qd


def ground_dynamic_friction_delta(body_q: torch.Tensor, body_qd: torch.Tensor,
                                  r: torch.Tensor, n: torch.Tensor,
                                  m_inv: torch.Tensor, I_inv: torch.Tensor,
                                  dynamic_friction: torch.Tensor,
                                  lambda_n: torch.Tensor, dt: float):
    dbody_qd = torch.zeros(6, device=body_q.device)  # Linear and angular correction

    q = body_q[3:]  # Orientation
    v = body_qd[3:]  # Linear velocity
    w = body_qd[:3]  # Angular velocity

    # Compute the relative velocity of the contact point in the world frame
    v_rel = v + rotate_vectors(torch.linalg.cross(w, r), q)

    # Compute the tangent component of the relative velocity
    v_t = v_rel - torch.dot(v_rel, n) * n
    v_t_magnitude = torch.norm(v_t)
    t = v_t / v_t_magnitude

    if v_t_magnitude < 1e-5:
        return dbody_qd

    # Compute the change of velocity due to the friction (now infinite friction)
    coulomb_friction = torch.abs(dynamic_friction * lambda_n / (dt**2))
    delta_v_friction = -v_t * torch.min(coulomb_friction,
                                        v_t_magnitude) / v_t_magnitude

    # Compute the impulse due to the friction in the world frame
    tb = rotate_vectors_inverse(t, q)  # Rotate the tangent to the body frame
    J_friction = delta_v_friction / (m_inv + torch.dot(
        torch.linalg.cross(I_inv @ torch.linalg.cross(r, tb), r), tb))
    dv = J_friction * m_inv
    dw = I_inv @ torch.linalg.cross(r, rotate_vectors_inverse(J_friction, q))

    dbody_qd = torch.cat([dw, dv])
    return dbody_qd


def revolute_joint_angular_delta(body_q_p: torch.Tensor,
                                 body_q_c: torch.Tensor, X_p: torch.Tensor,
                                 X_c: torch.Tensor, joint_axis: torch.Tensor,
                                 I_p_inv: torch.Tensor, I_c_inv: torch.Tensor):
    q_p = body_q_p[3:]
    q_c = body_q_c[3:]

    X_wj_p = transform_multiply(body_q_p, X_p)
    X_wj_c = transform_multiply(body_q_c, X_c)

    axis_w_p = rotate_vectors(joint_axis, X_wj_p[3:])
    axis_w_c = rotate_vectors(joint_axis, X_wj_c[3:])

    rot_vector = torch.linalg.cross(axis_w_p, axis_w_c)
    theta = torch.linalg.norm(rot_vector)

    if theta < 1e-6:
        return torch.zeros(4), torch.zeros(4)

    n = rot_vector / theta
    n_p = rotate_vectors_inverse(n, q_p)
    n_c = rotate_vectors_inverse(n, q_c)

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


def joint_delta(body_q_p: torch.Tensor, body_q_c: torch.Tensor,
                X_p: torch.Tensor, X_c: torch.Tensor, joint_axis: torch.Tensor,
                m_p_inv: torch.Tensor, m_c_inv: torch.Tensor,
                I_p_inv: torch.Tensor, I_c_inv: torch.Tensor):

    dq_p, dq_c = revolute_joint_angular_delta(body_q_p, body_q_c, X_p, X_c,
                                              joint_axis, I_p_inv, I_c_inv)

    new_body_q_p = body_q_p.clone()
    new_body_q_c = body_q_c.clone()
    new_body_q_p[3:] = normalize_quat(new_body_q_p[3:] + dq_p)
    new_body_q_c[3:] = normalize_quat(new_body_q_c[3:] + dq_c)

    dbody_q_p, dbody_q_c, _ = positional_delta(body_q_a=new_body_q_p,
                                               body_q_b=new_body_q_c,
                                               r_a=X_p[:3],
                                               r_b=X_c[:3],
                                               m_a_inv=m_p_inv,
                                               m_b_inv=m_c_inv,
                                               I_a_inv=I_p_inv,
                                               I_b_inv=I_c_inv)
    dbody_q_p[3:] = dbody_q_p[3:] + dq_p
    dbody_q_c[3:] = dbody_q_c[3:] + dq_c

    return dbody_q_p, dbody_q_c
