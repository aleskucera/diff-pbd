import torch
from pbd_torch.collision import *
from pbd_torch.correction import *
from pbd_torch.model import *
from pbd_torch.transform import *


def resolve_collisions(model: Model, state: State):
    robot_dx = torch.zeros(3)
    robot_dq = torch.zeros(4)

    print()
    state.robot_contact_deltas = torch.zeros((state.contact_count, 7))

    for i in range(state.contact_count):
        b = state.contact_body[i].item()  # Body index
        r = state.contact_point[i]  # Contact point in body coordinates

        robot_x = state.robot_q[:3]  # Robot position
        robot_rot = state.robot_q[3:]  # Robot rotation
        robot_m_inv = model.robot_inv_mass
        robot_I_inv = model.robot_inv_inertia

        r_a_world = body_to_world(r, state.body_q[b])
        r_b_world = torch.tensor([r_a_world[0], r_a_world[1], 0.0])

        if (r_a_world[2] >= 0):
            continue

        r_a_robot = body_to_robot(r, state.body_q[b], state.robot_q)

        dx, _, dq, _, d_lambda = positional_delta(x_a=robot_x,
                                                  x_b=torch.tensor(
                                                      [0.0, 0.0, 0.0]),
                                                  q_a=robot_rot,
                                                  q_b=ROT_IDENTITY,
                                                  r_a=r_a_robot,
                                                  r_b=r_b_world,
                                                  m_a_inv=robot_m_inv,
                                                  m_b_inv=0.0,
                                                  I_a_inv=robot_I_inv,
                                                  I_b_inv=torch.zeros(3, 3))
        robot_dx += dx
        robot_dq += dq
        print(f"\t\t[{model.body_name[b]}] dx: {[f'{x:.3f}' for x in dx.tolist()]}")
        print(
            f"\t\t[{model.body_name[b]}] dq: {[f'{x:.3f}' for x in dq.tolist()]}"
        )
        state.robot_contact_deltas[i] = torch.cat([dx, dq])

    print(f"\t Robot dx: {[f'{x:.3f}' for x in robot_dx.tolist()]}")
    print(
        f"\t Robot dq: {[f'{x:.3f}' for x in robot_dq.tolist()]}"
    )
    print()

def update_velocity(model: Model, state_prev: State, state: State, dt: float):
    x = state.robot_q[:3]  # Position
    q = state.robot_q[3:]  # Rotation

    x_prev = state_prev.robot_q[:3]  # Previous position
    q_prev = state_prev.robot_q[3:]  # Previous rotation

    v = (x - x_prev) / dt
    w = numerical_angular_velocity(q, q_prev, dt)

    print(f"\tRobot old qd: {state_prev.robot_qd}")
    print(f"\tRobot new qd: {torch.cat([w, v])}")

    state.robot_qd = torch.cat([w, v]).clone()


def numerical_angular_velocity(q: torch.Tensor, q_prev: torch.Tensor,
                               dt: float):
    q_prev_inv = quat_inv(q_prev)
    q_rel = quat_mul(q_prev_inv, q)
    omega = 2 * torch.tensor([q_rel[1], q_rel[2], q_rel[3]]) / dt
    if q_rel[0] < 0:
        omega = -omega
    return omega


def integrate_robot(model: Model, state_in: State, state_out: State,
                    dt: float):
    m_inv = model.robot_inv_mass
    I_inv = model.robot_inv_inertia

    x0 = state_in.robot_q[:3]  # Position
    r0 = state_in.robot_q[3:]  # Rotation
    v0 = state_in.robot_qd[3:]  # Linear velocity
    w0 = state_in.robot_qd[:3]  # Angular velocity

    t0 = state_in.robot_f[:3]  # Torque
    f0 = state_in.robot_f[3:]  # Linear force
    c = torch.linalg.cross(w0, torch.matmul(I_inv, w0))  # Coriolis force

    v1 = v0 + (f0 * m_inv + model.gravity) * dt
    x1 = x0 + v1 * dt

    w1 = w0 + torch.matmul(I_inv, t0 - c) * dt

    r1 = r0 + 0.5 * quat_mul(r0, torch.cat([torch.tensor([0.0]), w1])) * dt
    r1 = normalize_quat(r1)

    state_out.robot_q = torch.cat([x1, r1]).clone()
    state_out.robot_qd = torch.cat([w1, v1]).clone()


def eval_joint_force(q: float, qd: float, act: float, ke: float, kd: float,
                     mode: int) -> float:
    if mode == JOINT_MODE_FORCE:
        return act
    elif mode == JOINT_MODE_TARGET_POSITION:
        return ke * (act - q) - kd * (qd)
    elif mode == JOINT_MODE_TARGET_VELOCITY:
        return ke * (act - qd)
    else:
        raise ValueError("Invalid joint mode")


def add_control(model: Model, state: State, control: Control):
    state.joint_f = torch.zeros((model.joint_count))
    for i in range(model.joint_count):
        q = state.joint_q[i]
        qd = state.joint_qd[i]
        ke = model.joint_ke[i]
        kd = model.joint_kd[i]
        act = control.joint_act[i]
        mode = model.joint_axis_mode[i]
        force = eval_joint_force(q, qd, act, ke, kd, mode)

        state.joint_f[i] = force


def robot_integrate_joints(model: Model, state_in: State, state_out: State,
                           dt: float):
    for j in range(model.joint_count):
        f = state_in.joint_f[j]
        wheel = model.joint_child[j]
        joint_axis = model.joint_axis[j]
        I_inv = model.body_inv_inertia[wheel]

        torque = f * joint_axis

        state_in.body_qd[wheel][:3] += torch.matmul(I_inv, torque) * dt


def robot_friction_force(model: Model, state: State):
    total_torque = torch.zeros(3)
    total_lin_force = torch.zeros(3)
    for i in range(state.contact_count):
        b = state.contact_body[i].item()
        n = state.contact_normal[i]
        r = state.contact_point[i]

        # Get the velocity of the contact point
        q = state.body_q[b, 3:]  # Rotation
        v = state.body_qd[b, 3:]  # Linear velocity
        w = state.body_qd[b, :3]  # Angular velocity

        v_point = v + rotate_vectors(torch.linalg.cross(w, r), q)  # Velocity of the contact point
        v_normal = torch.dot(v_point, n) * n
        v_tangent = v_point - v_normal

        mu = 0.5
        friction_force = -mu * v_tangent

        # Compute the linear force due to the friction
        total_lin_force += friction_force

        # Compute the torque due to the friction
        r_robot = body_to_robot(r, state.body_q[b], state.robot_q)
        torque = torch.cross(r_robot, friction_force)
        total_torque += torque

    if state.contact_count > 0:
        total_torque /= state.contact_count
        total_lin_force /= state.contact_count
    state.robot_f = torch.cat([total_torque, total_lin_force])


def update_bodies(model: Model, state: State):
    print(f"\tRobot q: {state.robot_q}")
    for b in range(model.body_count):
        robot_x = state.robot_q[:3]
        robot_rot = state.robot_q[3:]

        init_robot_x = model.robot_q[:3]
        init_robot_rot = model.robot_q[3:]

        init_body_x = model.body_q[b, :3]
        init_body_rot = model.body_q[b, 3:]

        # q_rel = relative_rotation(robot_rot, init_robot_rot)
        q_rel = relative_rotation(init_robot_rot, robot_rot)

        body_x = robot_x + rotate_vectors(init_body_x - init_robot_x, q_rel)
        body_rot = quat_mul(q_rel, init_body_rot)

        print(f"\tBody {model.body_name[b]}:")
        print(
            f"\t\tOld body_q: X = {[f'{x:.3f}' for x in state.body_q[b][:3].tolist()]}, "
            f"R = {[f'{x:.3f}' for x in quat_to_rotvec(state.body_q[b][3:]).tolist()]}\n"
            f"\t\tNew body_q: X = {[f'{x:.3f}' for x in body_x.tolist()]}, "
            f"R = {[f'{x:.3f}' for x in quat_to_rotvec(body_rot).tolist()]}")

        state.body_q[b] = torch.cat([body_x, body_rot])


class RobotIntegrator:

    def __init__(self, iterations: int = 2):
        self.iterations = iterations

    def simulate(self, model: Model, state_in: State, state_out: State,
                 control: Control, dt: float):
        print()
        print(f"=============== TIME: {state_in.time:.3f} ===============")
        integrate_robot(model, state_in, state_out, dt)
        update_bodies(model, state_out)
        collide(model, state_out)
        for _ in range(self.iterations):
            resolve_collisions(model, state_out)
            state_out.add_robot_contact_deltas()
            update_bodies(model, state_out)
        update_velocity(model, state_in, state_out, dt)

        # update_bodies(model, state_out)
        # collide(model, state_out)
        # robot_friction_force(model, state_out)
        # integrate_robot(model, state_in, state_out, dt)
        # collide(model, state_in)
        # add_control(model, state_in, control)
        # robot_integrate_joints(model, state_in, state_in, dt)
        # robot_friction_force(model, state_out)
        # integrate_robot(model, state_in, state_out, dt)
        # for _ in range(self.iterations):
        #     update_bodies(model, state_out)
        #     collide(model, state_out)
        #     resolve_collisions(model, state_out)
        # update_velocity(model, state_in, state_out, dt)
