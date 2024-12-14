from typing import Tuple

import matplotlib
import torch
from pbd_torch.transform import *
from pbd_torch.utils import *

matplotlib.use('TkAgg')


def numerical_angular_velocity(q: torch.Tensor, q_prev: torch.Tensor,
                               dt: float):
    q_prev_inv = quat_inv(q_prev)
    q_rel = quat_mul(q_prev_inv, q)
    omega = 2 * torch.tensor([q_rel[1], q_rel[2], q_rel[3]]) / dt
    if q_rel[0] < 0:
        omega = -omega
    return omega


def numerical_linear_velocity(x: torch.Tensor, x_prev: torch.Tensor,
                              dt: float):
    return (x - x_prev) / dt


def velocity_update(x: torch.Tensor, x_prev: torch.Tensor, q: torch.Tensor,
                    q_prev: torch.Tensor, dt: float):
    v = numerical_linear_velocity(x, x_prev, dt)
    w = numerical_angular_velocity(q, q_prev, dt)
    return v, w


def restitution_delta(q: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
                      v_prev: torch.Tensor, w_prev: torch.Tensor,
                      r: torch.Tensor, n: torch.Tensor, m_inv: float,
                      I_inv: torch.Tensor, restitution: float):
    # Compute the normal component of the relative velocity
    v_rel = v + rotate_vectors(torch.linalg.cross(w, r), q)
    v_n_magnitude = torch.dot(v_rel, n)
    v_n = v_n_magnitude * n

    if torch.abs(v_n_magnitude) < 1e-5:
        restitution = 0.0

    # Compute the normal component of the relative velocity before the velocity update
    v_rel_prev = v_prev + rotate_vectors(torch.linalg.cross(w_prev, r), q)
    v_n_magnitude_prev = torch.dot(v_rel_prev, n)
    v_n_prev = v_n_magnitude_prev * n

    # Compute the change of velocity due to the restitution
    delta_v_restitution = -(v_n + restitution * v_n_prev)

    # Compute the impulse due to the restitution in the world frame
    nb = rotate_vectors_inverse(n, q)  # Rotate the normal to the body frame
    J_restitution = delta_v_restitution / (m_inv + torch.dot(
        torch.linalg.cross(I_inv @ torch.linalg.cross(r, nb), r), nb))
    dv = J_restitution * m_inv
    dw = I_inv @ torch.linalg.cross(r, rotate_vectors_inverse(
        J_restitution, q))

    return dv, dw


def dynamic_friction_delta(q: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
                           r: torch.Tensor, n: torch.Tensor, m_inv: float,
                           I_inv: torch.Tensor, dt: float,
                           dynamic_friction: float, lambda_n: torch.Tensor):
    # Compute the relative velocity of the contact point in the world frame
    v_rel = v + rotate_vectors(torch.linalg.cross(w, r), q)

    # Compute the tangent component of the relative velocity
    v_t = v_rel - torch.dot(v_rel, n) * n
    v_t_magnitude = torch.norm(v_t)
    t = v_t / v_t_magnitude

    # Compute the change of velocity due to the friction (now infinite friction)
    coulomb_friction = torch.abs(dynamic_friction * lambda_n / dt)
    delta_v_friction = -v_t  # * torch.min(coulomb_friction, v_t_magnitude) / v_t_magnitude

    # Compute the impulse due to the friction in the world frame
    tb = rotate_vectors_inverse(t, q)  # Rotate the tangent to the body frame
    J_friction = delta_v_friction / (m_inv + torch.dot(
        torch.linalg.cross(I_inv @ torch.linalg.cross(r, tb), r), tb))
    dv = J_friction * m_inv
    dw = I_inv @ torch.linalg.cross(r, rotate_vectors_inverse(J_friction, q))

    return dv, dw


def positional_delta(
    x_a: torch.Tensor, x_b: torch.Tensor, q_a: torch.Tensor, q_b: torch.Tensor,
    r_a: torch.Tensor, r_b: torch.Tensor, m_a_inv: float, m_b_inv: float,
    I_a_inv: torch.Tensor, I_b_inv: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    dx_a = torch.zeros(3)
    dx_b = torch.zeros(3)
    dq_a = torch.zeros(4)
    dq_b = torch.zeros(4)
    d_lambda = torch.tensor(0.0)

    delta_x = rotate_vectors(r_a, q_a) + x_a - rotate_vectors(r_b, q_b) - x_b

    c = torch.norm(delta_x)
    if c < 1e-6:
        return dx_a, dx_b, dq_a, dq_b, d_lambda

    n = delta_x / c

    n_a = rotate_vectors_inverse(n, q_a)
    n_b = rotate_vectors_inverse(n, q_b)

    weight_a = m_a_inv + torch.dot(
        torch.linalg.cross(I_a_inv @ torch.linalg.cross(r_a, n_a), r_a), n_a)
    weight_b = m_b_inv + torch.dot(
        torch.linalg.cross(I_b_inv @ torch.linalg.cross(r_b, n_b), r_b), n_b)

    d_lambda = -c / (weight_a + weight_b)

    p = d_lambda * n
    p_a = d_lambda * n_a
    p_b = d_lambda * n_b

    # Positional correction
    dx_a = p * m_a_inv
    dx_b = -p * m_b_inv

    # Rotational correction
    w_a = I_a_inv @ torch.linalg.cross(r_a, p_a)
    w_a_quat = torch.cat([torch.tensor([0.0]), w_a], dim=0)
    dq_a = 0.5 * quat_mul(q_a, w_a_quat)

    w_b = I_b_inv @ torch.linalg.cross(r_b, p_b)
    w_b_quat = torch.cat([torch.tensor([0.0]), w_b], dim=0)
    dq_b = -0.5 * quat_mul(q_b, w_b_quat)

    return dx_a, dx_b, dq_a, dq_b, d_lambda


def angular_delta(u_a: torch.Tensor, u_b: torch.Tensor, q_a: torch.Tensor,
                  q_b: torch.Tensor, I_a_inv: torch.Tensor,
                  I_b_inv: torch.Tensor):
    # Normalize the u_a and u_b vectors
    u_a = u_a / torch.linalg.norm(u_a)
    u_b = u_b / torch.linalg.norm(u_b)

    # Rotate the vectors to the world frame
    u_a_w = rotate_vectors(u_a, q_a)
    u_b_w = rotate_vectors(u_b, q_b)

    rot_vector = torch.linalg.cross(u_a_w, u_b_w)
    theta = torch.linalg.norm(rot_vector)

    if theta < 1e-6:
        return torch.zeros(4), torch.zeros(4)

    n = rot_vector / theta

    n_a = rotate_vectors_inverse(n, q_a)
    n_b = rotate_vectors_inverse(n, q_b)

    w1 = n_a @ I_a_inv @ n_a
    w2 = n_a @ I_b_inv @ n_b
    w = w1 + w2

    theta_a = theta * w1 / w
    theta_b = -theta * w2 / w

    q_a_correction = torch.cat([torch.tensor([0.0]), theta_a * n_a], dim=0)
    q_b_correction = torch.cat([torch.tensor([0.0]), theta_b * n_b], dim=0)

    dq_a = 0.5 * quat_mul(q_a, q_a_correction)
    dq_b = 0.5 * quat_mul(q_b, q_b_correction)

    return dq_a, dq_b
