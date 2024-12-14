import torch
from pbd_torch.collision import *
from pbd_torch.correction import *
from pbd_torch.model import *
from pbd_torch.transform import *


def integrate_bodies(model: Model, state_in: State, state_out: State,
                     dt: float):
    for b in range(model.body_count):
        m_inv = model.body_inv_mass[b]
        I_inv = model.body_inv_inertia[b]

        x0 = state_in.body_q[b, :3]  # Position
        r0 = state_in.body_q[b, 3:]  # Rotation
        v0 = state_in.body_qd[b, 3:]  # Linear velocity
        w0 = state_in.body_qd[b, :3]  # Angular velocity

        t0 = state_in.body_f[b, :3]  # Torque
        f0 = state_in.body_f[b, 3:]  # Linear force
        c = torch.linalg.cross(w0, torch.matmul(I_inv, w0))  # Coriolis force

        v1 = v0 + (f0 * m_inv + model.gravity) * dt
        x1 = x0 + v1 * dt

        w1 = w0 + torch.matmul(I_inv, t0 - c) * dt

        r1 = r0 + 0.5 * quat_mul(r0, torch.cat([torch.tensor([0.0]), w1])) * dt
        r1 = normalize_quat(r1)

        state_out.body_q[b] = torch.cat([x1, r1]).clone()
        state_out.body_qd[b] = torch.cat([w1, v1]).clone()


def numerical_angular_velocity(q: torch.Tensor, q_prev: torch.Tensor,
                               dt: float):
    q_prev_inv = quat_inv(q_prev)
    q_rel = quat_mul(q_prev_inv, q)
    omega = 2 * torch.tensor([q_rel[1], q_rel[2], q_rel[3]]) / dt
    if q_rel[0] < 0:
        omega = -omega
    return omega


def update_velocity(model: Model, state_prev: State, state: State, dt: float):
    for b in range(model.body_count):
        x = state.body_q[b, :3]  # Position
        q = state.body_q[b, 3:]  # Rotation

        x_prev = state_prev.body_q[b, :3]  # Previous position
        q_prev = state_prev.body_q[b, 3:]  # Previous rotation

        v = (x - x_prev) / dt
        w = numerical_angular_velocity(q, q_prev, dt)

        state.body_qd[b] = torch.cat([w, v]).clone()


def ground_contact_deltas(model: Model, state: State):
    x_deltas = torch.zeros((state.contact_count, 3))
    q_deltas = torch.zeros((state.contact_count, 4))
    lambda_deltas = torch.zeros(state.contact_count)

    for i in range(state.contact_count):
        b = state.contact_body[i].item(
        )  # Body index which is in contact with ground
        r = state.contact_point[i]  # Contact point in body coordinates

        x = state.body_q[b, :3]  # Body position
        q = state.body_q[b, 3:]  # Body rotation
        m_inv = model.body_inv_mass[b]
        I_inv = model.body_inv_inertia[b]

        r_a_world = rotate_vectors(r, q) + x
        r_b_world = torch.tensor([r_a_world[0], r_a_world[1], 0.0])

        if r_a_world[2] >= 0:
            continue

        dx, _, dq, _, d_lambda = positional_delta(x_a=x,
                                                  x_b=torch.tensor(
                                                      [0.0, 0.0, 0.0]),
                                                  q_a=q,
                                                  q_b=ROT_IDENTITY,
                                                  r_a=r,
                                                  r_b=r_b_world,
                                                  m_a_inv=m_inv,
                                                  m_b_inv=0.0,
                                                  I_a_inv=I_inv,
                                                  I_b_inv=torch.zeros(3, 3))

        x_deltas[i] = dx
        q_deltas[i] = dq
        lambda_deltas[i] = d_lambda

    state.contact_deltas = torch.cat((x_deltas, q_deltas), dim=1)

    return x_deltas, q_deltas, lambda_deltas


def restitution_deltas(model: Model, state: State, state_prev: State):
    v_deltas = torch.zeros((state.contact_count, 3))
    w_deltas = torch.zeros((state.contact_count, 3))

    for i in range(state.contact_count):
        b = state.contact_body[i].item(
        )  # Body index which is in contact with ground
        r = state.contact_point[i]  # Contact point in body coordinates

        n = state.contact_normal[i]  # Contact normal in world coordinates

        q = state.body_q[b, 3:]  # Body rotation
        v = state.body_qd[b, 3:]  # Linear velocity
        w = state.body_qd[b, :3]  # Angular velocity

        v_prev = state_prev.body_qd[b, 3:]  # Previous linear velocity
        w_prev = state_prev.body_qd[b, :3]  # Previous angular velocity

        m_inv = model.body_inv_mass[b]  # Inverse mass
        I_inv = model.body_inv_inertia[b]  # Inverse inertia

        dv, dw = restitution_delta(q, v, w, v_prev, w_prev, r, n, m_inv, I_inv,
                                   0.5)
        v_deltas[i] = dv
        w_deltas[i] = dw

    return v_deltas, w_deltas


class XPBDIntegrator:

    def __init__(self, iterations: int = 2):
        self.iterations = iterations

    def simulate(self, model: Model, state_in: State, state_out: State,
                 control: Control, dt: float):
        integrate_bodies(model, state_in, state_out, dt)
        collide(model, state_out)
        for _ in range(self.iterations):
            ground_contact_deltas(model, state_out)
            state_out.add_contact_deltas()
        update_velocity(model, state_in, state_out, dt)
