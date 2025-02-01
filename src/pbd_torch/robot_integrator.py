import torch
from pbd_torch.collision import *
from pbd_torch.correction import *
from pbd_torch.integrator import integrate_body
from pbd_torch.model import *
from pbd_torch.transform import *


def update_body_with_respect_to_robot(body_q: torch.Tensor,
                                      init_body_q: torch.Tensor,
                                      robot_q: torch.Tensor,
                                      init_robot_q: torch.Tensor):
    robot_x = robot_q[:3]
    robot_rot = robot_q[3:]

    init_robot_x = init_robot_q[:3]
    init_robot_rot = init_robot_q[3:]

    init_body_x = init_body_q[:3]
    init_body_rot = init_body_q[3:]

    q_rel = relative_rotation(init_robot_rot, robot_rot)

    body_x = robot_x + rotate_vectors(init_body_x - init_robot_x, q_rel)
    body_rot = quat_mul(q_rel, init_body_rot)

    new_body_q = torch.cat([body_x, body_rot])
    return new_body_q


# def integrate_body(body_q: torch.Tensor, body_qd: torch.Tensor,
#                    body_f: torch.Tensor, body_inv_mass: float,
#                    body_inv_inertia: torch.Tensor, gravity: torch.Tensor,
#                    dt: float):
#     x0 = body_q[:3]  # Position
#     q0 = body_q[3:]  # Rotation
#     v0 = body_qd[3:]  # Linear velocity
#     w0 = body_qd[:3]  # Angular velocity
#     t0 = body_f[:3]  # Torque
#     f0 = body_f[3:]  # Linear force
#     c = torch.linalg.cross(w0, torch.matmul(body_inv_inertia,
#                                             w0))  # Coriolis force
#     v1 = v0 + (f0 * body_inv_mass + gravity) * dt
#     x1 = x0 + v1 * dt
#     w1 = w0 + torch.matmul(body_inv_inertia, t0 - c) * dt
#     q1 = q0 + 0.5 * quat_mul(q0, torch.cat([torch.tensor([0.0]), w1])) * dt
#     q1 = normalize_quat(q1)
#
#     new_body_q = torch.cat([x1, q1])
#     new_body_qd = torch.cat([w1, v1])
#
#     return new_body_q, new_body_qd


def numerical_qd(q: torch.Tensor, q_prev: torch.Tensor, dt: float):
    x = q[:3]
    q = q[3:]

    x_prev = q_prev[:3]
    q_prev = q_prev[3:]

    # Compute the linear velocity
    v = (x - x_prev) / dt

    # Compute the angular velocity
    q_rel = quat_mul(quat_inv(q_prev), q)
    omega = 2 * torch.tensor([q_rel[1], q_rel[2], q_rel[3]]) / dt
    if q_rel[0] < 0:
        omega = -omega

    qd = torch.cat([omega, v])
    return qd


def robot_ground_contact_deltas(robot_q: torch.Tensor, body_q: torch.Tensor,
                                contact_count: int, contact_body: torch.Tensor,
                                contact_point: torch.Tensor,
                                robot_inv_mass: torch.Tensor,
                                robot_inv_inertia: torch.Tensor):
    lambda_ = 0.0
    contact_deltas = torch.zeros((contact_count, 7))

    for i in range(contact_count):
        b = contact_body[i].item()
        r = contact_point[i]

        r_a_world = body_to_world(r, body_q[b])
        r_b_world = torch.tensor([r_a_world[0], r_a_world[1], 0.0])

        if (r_a_world[2] >= 0):
            continue

        r_a_robot = body_to_robot(r, body_q[b], robot_q)

        dbody_q, _, d_lambda = positional_delta(body_q_a=robot_q,
                                                body_q_b=TRANSFORM_IDENTITY,
                                                r_a=r_a_robot,
                                                r_b=r_b_world,
                                                m_a_inv=robot_inv_mass,
                                                m_b_inv=torch.zeros(1),
                                                I_a_inv=robot_inv_inertia,
                                                I_b_inv=torch.zeros(3, 3))
        lambda_ += d_lambda
        contact_deltas[i] = dbody_q

    return contact_deltas, lambda_


def robot_friction_force(body_q: torch.Tensor, body_qd: torch.Tensor,
                         robot_q: torch.Tensor, contact_count: int,
                         contact_body: torch.Tensor,
                         contact_point: torch.Tensor,
                         contact_normal: torch.Tensor, friction_coeff: float):
    total_torque = torch.zeros(3)
    total_lin_force = torch.zeros(3)
    for i in range(contact_count):
        b = contact_body[i].item()
        n = contact_normal[i]
        r = contact_point[i]

        # Get the velocity of the contact point
        q = body_q[b, 3:]  # Rotation
        v = body_qd[b, 3:]  # Linear velocity
        w = body_qd[b, :3]  # Angular Velocity

        # Compute the relative velocity of the contact point
        v_point = v + rotate_vectors(torch.linalg.cross(w, r), q)

        # Get the normal and tangent components of the velocity
        v_normal = torch.dot(v_point, n) * n
        v_tangent = v_point - v_normal

        friction_force = -friction_coeff * v_tangent

        # Compute the linear force due to the friction
        total_lin_force += friction_force

        # Compute the torque due to the friction
        r_robot = body_to_robot(r, body_q[b], robot_q)
        torque = torch.linalg.cross(r_robot, friction_force)
        total_torque += torque

    if contact_count > 0:
        total_torque /= contact_count
        total_lin_force /= contact_count

    total_force = torch.cat([total_torque, total_lin_force])
    return total_force


class RobotIntegrator:

    def __init__(self, iterations: int = 2):
        self.iterations = iterations

    def simulate(self, model: Model, state_in: State, state_out: State,
                 control: Control, dt: float):

        # Get the init state
        body_q = state_in.body_q.clone()
        body_qd = state_in.body_qd.clone()
        robot_q = state_in.robot_q.clone()
        robot_qd = state_in.robot_qd.clone()
        robot_f = state_in.robot_f.clone()

        # Save the contact points to the state_in
        state_in.contact_count = model.contact_count
        state_in.contact_body = model.contact_body.clone()
        state_in.contact_point = model.contact_point.clone()
        state_in.contact_normal = model.contact_normal.clone()
        state_in.contact_point_idx = model.contact_point_idx.clone()

        # Compute the friction force for the robot
        friction_force = robot_friction_force(body_q,
                                              body_qd,
                                              robot_q,
                                              model.contact_count,
                                              model.contact_body,
                                              model.contact_point,
                                              model.contact_normal,
                                              friction_coeff=0.2)
        robot_f += friction_force

        state_in.robot_f = robot_f.clone()

        # Integrate the robot
        robot_q, robot_qd = integrate_body(robot_q, robot_qd, robot_f,
                                           model.robot_inv_mass,
                                           model.robot_inv_inertia,
                                           model.gravity, dt)

        # Update the bodies with respect to the robot
        for i in range(model.body_count):
            body_q[i] = update_body_with_respect_to_robot(
                body_q=body_q[i],
                init_body_q=model.body_q[i],
                robot_q=robot_q,
                init_robot_q=model.robot_q)

        # Update positions to resolve collisions
        n_lambda = 0.0
        if model.contact_count > 0:
            for _ in range(self.iterations):
                robot_contact_deltas, d_lambda = robot_ground_contact_deltas(
                    robot_q, body_q, model.contact_count, model.contact_body,
                    model.contact_point, model.robot_inv_mass,
                    model.robot_inv_inertia)

                # Add the lambda to the normal lambda (normal force)
                n_lambda += d_lambda

                # Check how many contact correction were made
                non_zero_mask = torch.any(robot_contact_deltas != 0, dim=1)
                num_non_zero = torch.sum(non_zero_mask).item()
                if num_non_zero == 0:
                    break

                # Update the robot position and rotation
                delta_q = torch.sum(robot_contact_deltas, dim=0) / num_non_zero
                robot_q[:3] += delta_q[:3]
                robot_q[3:] = normalize_quat(robot_q[3:] + delta_q[3:])

                for i in range(model.body_count):
                    body_q[i] = update_body_with_respect_to_robot(
                        body_q=body_q[i],
                        init_body_q=model.body_q[i],
                        robot_q=robot_q,
                        init_robot_q=model.robot_q)

        # Update the velocities
        robot_qd = numerical_qd(robot_q, state_in.robot_q, dt)

        # Save the output state
        state_out.robot_q = robot_q
        state_out.robot_qd = robot_qd
        state_out.body_q = body_q
        state_out.body_qd = body_qd
        state_out.time = state_in.time + dt
