import os
from typing import Tuple

import torch
from pbd_torch.collision import *
from pbd_torch.correction import *
from pbd_torch.model import *
from pbd_torch.printer import DebugPrinter
from pbd_torch.transform import *

os.environ['DEBUG'] = 'false'


def integrate_body(body_q: torch.Tensor, body_qd: torch.Tensor,
                   body_f: torch.Tensor, body_inv_mass: torch.Tensor,
                   body_inv_inertia: torch.Tensor, gravity: torch.Tensor,
                   dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integrates the body's position and orientation over a time step.

    Args:
        body_q (torch.Tensor): Current body state (position[3] and quaternion[4])
        body_qd (torch.Tensor): Current body velocities (angular[3] and linear[3])
        body_f (torch.Tensor): Forces and torques applied to the body
        body_inv_mass (torch.Tensor): Inverse mass of the body
        body_inv_inertia (torch.Tensor): Inverse inertia tensor
        gravity (torch.Tensor): Gravity vector
        dt (float): Time step

    Returns:
        tuple: Updated (body_q, body_qd)
    """

    x0 = body_q[:3].clone()  # Position
    q0 = body_q[3:].clone()  # Rotation
    v0 = body_qd[3:].clone()  # Linear velocity
    w0 = body_qd[:3].clone()  # Angular velocity
    t0 = body_f[:3].clone()  # Torque
    f0 = body_f[3:].clone()  # Linear force
    c = torch.linalg.cross(w0, body_inv_inertia @ w0)  # Coriolis force

    # Integrate the velocity and position
    v1 = v0 + (f0 * body_inv_mass + gravity) * dt
    x1 = x0 + v1 * dt

    # Integrate the angular velocity and orientation
    w1 = w0 + torch.matmul(body_inv_inertia, t0 - c) * dt
    w1_w = rotate_vectors(w1, q0)
    q1 = q0 + 0.5 * quat_mul(torch.cat([torch.tensor([0.0], device=w1_w.device), w1_w]), q0) * dt
    q1 = normalize_quat(q1)

    new_body_q = torch.cat([x1, q1])
    new_body_qd = torch.cat([w1, v1])

    return new_body_q, new_body_qd


def numerical_qd(body_q: torch.Tensor, body_q_prev: torch.Tensor,
                 dt: float) -> torch.Tensor:
    """Computes numerical velocities from positions at current and previous time steps.

    Args:
        body_q (torch.Tensor): Current body state
        body_q_prev (torch.Tensor): Previous body state
        dt (float): Time step

    Returns:
        torch.Tensor: Computed velocities
    """

    x, q = body_q[:3], body_q[3:]
    x_prev, q_prev = body_q_prev[:3], body_q_prev[3:]

    # Compute the linear velocity
    v = (x - x_prev) / dt

    # Compute the angular velocity
    q_rel = quat_mul(quat_inv(q_prev), q)
    omega = 2 * torch.tensor([q_rel[1], q_rel[2], q_rel[3]], device=q_rel.device) / dt
    if q_rel[0] < 0:
        omega = -omega

    qd = torch.cat([omega, v])
    return qd


def get_ground_contact_deltas(
        body_q: torch.Tensor, contact_count: int, contact_body: torch.Tensor,
        contact_point: torch.Tensor, body_inv_mass: torch.Tensor,
        body_inv_inertia: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes position corrections for ground contacts.

    Args:
        body_q (torch.Tensor): Body states
        contact_count (int): Number of contacts
        contact_body (torch.Tensor): Body indices for each contact
        contact_point (torch.Tensor): Contact points in body coordinates
        body_inv_mass (torch.Tensor): Inverse masses
        body_inv_inertia (torch.Tensor): Inverse inertia tensors

    Returns:
        tuple: (contact_deltas, lambda) Position corrections and Lagrange multiplier
    """
    device = body_q.device
    lambda_ = torch.tensor([0.0], device=device)
    contact_deltas = torch.zeros((contact_count, 7), device=device)

    for i in range(contact_count):
        b = contact_body[i].item()  # Body index
        r = contact_point[i]  # Contact point in body coordinates

        m_inv = body_inv_mass[b]
        I_inv = body_inv_inertia[b]

        r_a_world = body_to_world(r, body_q[b])
        r_b_world = torch.tensor([r_a_world[0], r_a_world[1], 0.0], device=device)
        body_q_b = TRANSFORM_IDENTITY.to(device)
        m_b_inv = torch.zeros(1, device=device)
        I_b_inv = torch.zeros(3, 3, device=device)

        if r_a_world[2] >= 0:
            continue

        dbody_q, _, d_lambda = positional_delta(body_q_a=body_q[b],
                                                body_q_b=body_q_b,
                                                r_a=r,
                                                r_b=r_b_world,
                                                m_a_inv=m_inv,
                                                m_b_inv=m_b_inv,
                                                I_a_inv=I_inv,
                                                I_b_inv=I_b_inv)
        lambda_ = lambda_ + d_lambda
        contact_deltas[i] = dbody_q

    return contact_deltas, lambda_


def get_restitution_deltas(body_q: torch.Tensor, body_qd: torch.Tensor,
                           body_qd_prev: torch.Tensor, contact_count: int,
                           contact_body: torch.Tensor,
                           contact_point: torch.Tensor,
                           contact_normal: torch.Tensor,
                           body_inv_mass: torch.Tensor,
                           body_inv_inertia: torch.Tensor,
                           restitution: torch.Tensor) -> torch.Tensor:
    """Computes velocity corrections for restitution handling.

    Args:
        body_q (torch.Tensor): Body states
        body_qd (torch.Tensor): Current velocities
        body_qd_prev (torch.Tensor): Previous velocities
        contact_count (int): Number of contacts
        contact_body (torch.Tensor): Body indices for each contact
        contact_point (torch.Tensor): Contact points
        contact_normal (torch.Tensor): Contact normals
        body_inv_mass (torch.Tensor): Inverse masses
        body_inv_inertia (torch.Tensor): Inverse inertia tensors
        restitution (torch.Tensor): Restitution coefficients

    Returns:
        torch.Tensor: Velocity corrections
    """
    restitution_deltas = torch.zeros((contact_count, 6), device=body_q.device)

    for c in range(contact_count):
        b = contact_body[c].item()  # Body index
        r = contact_point[c]  # Contact point in body coordinates

        r_a_world = body_to_world(r, body_q[b])

        if r_a_world[2] >= 0:
            continue

        dbody_qd = ground_restitution_delta(body_q=body_q[b],
                                            body_qd=body_qd[b],
                                            body_qd_prev=body_qd_prev[b],
                                            r=contact_point[c],
                                            n=contact_normal[c],
                                            m_inv=body_inv_mass[b],
                                            I_inv=body_inv_inertia[b],
                                            restitution=restitution[b])

        restitution_deltas[c] = dbody_qd

    return restitution_deltas


def get_dynamic_friction_deltas(body_q: torch.Tensor, body_qd: torch.Tensor,
                                contact_count: int, contact_body: torch.Tensor,
                                contact_point: torch.Tensor,
                                contact_normal: torch.Tensor,
                                body_inv_mass: torch.Tensor,
                                body_inv_inertia: torch.Tensor,
                                dynamic_friction: torch.Tensor,
                                lambda_n: torch.Tensor,
                                dt: float) -> torch.Tensor:
    """Computes velocity corrections for dynamic friction.

    Args:
        body_q (torch.Tensor): Body states
        body_qd (torch.Tensor): Current velocities
        contact_count (int): Number of contacts
        contact_body (torch.Tensor): Body indices for each contact
        contact_point (torch.Tensor): Contact points
        contact_normal (torch.Tensor): Contact normals
        body_inv_mass (torch.Tensor): Inverse masses
        body_inv_inertia (torch.Tensor): Inverse inertia tensors
        dynamic_friction (torch.Tensor): Dynamic friction coefficients
        lambda_n (torch.Tensor): Normal force Lagrange multiplier
        dt (float): Time step

    Returns:
        torch.Tensor: Friction velocity corrections
    """
    dynamic_friction_deltas = torch.zeros((contact_count, 6), device=body_q.device)

    for c in range(contact_count):
        b = contact_body[c].item()
        r = contact_point[c]  # Contact point in body coordinates

        r_a_world = body_to_world(r, body_q[b])

        if r_a_world[2] >= 0.1:
            continue

        dbody_qd = ground_dynamic_friction_delta(
            body_q=body_q[b],
            body_qd=body_qd[b],
            r=contact_point[c],
            n=contact_normal[c],
            m_inv=body_inv_mass[b],
            I_inv=body_inv_inertia[b],
            dynamic_friction=dynamic_friction[b],
            lambda_n=lambda_n,
            dt=dt)

        dynamic_friction_deltas[c] = dbody_qd

    return dynamic_friction_deltas


def get_joint_deltas(
        body_q: torch.Tensor, joint_parent: torch.Tensor,
        joint_child: torch.Tensor, joint_X_p: torch.Tensor,
        joint_X_c: torch.Tensor, joint_axis: torch.Tensor,
        body_inv_mass: torch.Tensor,
        body_inv_inertia: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes position corrections for joint constraints.

    Args:
        body_q (torch.Tensor): Body states
        joint_parent (torch.Tensor): Parent body indices
        joint_child (torch.Tensor): Child body indices
        joint_X_p (torch.Tensor): Parent joint transforms
        joint_X_c (torch.Tensor): Child joint transforms
        joint_axis (torch.Tensor): Joint axes
        body_inv_mass (torch.Tensor): Inverse masses
        body_inv_inertia (torch.Tensor): Inverse inertia tensors

    Returns:
        tuple: (joint_deltas, num_corrections)
    """

    num_joints = joint_parent.shape[0]
    joint_deltas = torch.zeros((body_q.shape[0], 7))
    num_corrections = torch.zeros(body_q.shape[0])

    for j in range(num_joints):
        parent = joint_parent[j]
        child = joint_child[j]
        if parent == -1:
            m_p_inv = torch.zeros(1, device=body_q.device)
            I_p_inv = torch.zeros(3, 3, device=body_q.device)
            body_q_p = TRANSFORM_IDENTITY.to(body_q.device)
        else:
            m_p_inv = body_inv_mass[parent]
            I_p_inv = body_inv_inertia[parent]
            body_q_p = body_q[parent]
        dbody_q_p, dbody_q_c = joint_delta(body_q_p=body_q_p,
                                           body_q_c=body_q[child],
                                           X_p=joint_X_p[j],
                                           X_c=joint_X_c[j],
                                           joint_axis=joint_axis[j],
                                           m_p_inv=m_p_inv,
                                           m_c_inv=body_inv_mass[child],
                                           I_p_inv=I_p_inv,
                                           I_c_inv=body_inv_inertia[child])
        joint_deltas[parent] = joint_deltas[parent] + dbody_q_p
        num_corrections[parent] += 1

        joint_deltas[child] = joint_deltas[child] + dbody_q_c
        num_corrections[child] += 1

    return joint_deltas, num_corrections


def forces_from_joint_actions(body_q: torch.Tensor, joint_parent: torch.Tensor,
                              joint_child: torch.Tensor,
                              joint_X_p: torch.Tensor, joint_X_c: torch.Tensor,
                              joint_axis: torch.Tensor,
                              joint_axis_mode: torch.Tensor,
                              joint_act: torch.Tensor) -> torch.Tensor:
    """Computes forces from joint actuations.

    Args:
        body_q (torch.Tensor): Body states
        joint_parent (torch.Tensor): Parent body indices
        joint_child (torch.Tensor): Child body indices
        joint_X_p (torch.Tensor): Parent joint transforms
        joint_X_c (torch.Tensor): Child joint transforms
        joint_axis (torch.Tensor): Joint axes
        joint_axis_mode (torch.Tensor): Joint actuation modes
        joint_act (torch.Tensor): Joint actuation values

    Returns:
        torch.Tensor: Computed forces for each body
    """

    num_joints = joint_parent.shape[0]
    body_f = torch.zeros((body_q.shape[0], 6), dtype=torch.float32, device=body_q.device)

    for j in range(num_joints):
        parent = joint_parent[j]
        child = joint_child[j]
        X_p = joint_X_p[j]
        axis = joint_axis[j]
        act = joint_act[j]
        body_q_p = body_q[parent]
        body_q_c = body_q[child]

        # TODO: Resolve the joint actions (now just for forces)
        t = act

        X_wj_p = transform_multiply(body_q_p, X_p)
        axis_w = rotate_vectors(axis, X_wj_p[3:])

        joint_torque = axis_w * t

        body_f[parent, :3] = body_f[parent, :3] - rotate_vectors_inverse(joint_torque,
                                                     body_q_p[3:])
        body_f[child, :3] = body_f[child, :3] + rotate_vectors_inverse(joint_torque, body_q_c[3:])

    return body_f


def eval_joint_force(q: float, qd: float, act: float, ke: float, kd: float,
                     mode: int) -> float:
    """Evaluates joint force based on the joint mode.

    Args:
        q (float): Joint position
        qd (float): Joint velocity
        act (float): Joint actuation
        ke (float): Position gain
        kd (float): Velocity gain
        mode (int): Joint mode (FORCE, TARGET_POSITION, or TARGET_VELOCITY)

    Returns:
        float: Computed joint force
    """
    if mode == JOINT_MODE_FORCE:
        return act
    elif mode == JOINT_MODE_TARGET_POSITION:
        return ke * (act - q) - kd * qd
    elif mode == JOINT_MODE_TARGET_VELOCITY:
        return ke * (act - qd)
    else:
        raise ValueError("Invalid joint mode")


class XPBDIntegrator:

    def __init__(self, iterations: int = 2):
        self.iterations = iterations

    def simulate(self, model: Model, state_in: State, state_out: State,
                 control: Control, dt: float):
        debug = DebugPrinter()
        debug.section(f"TIME: {state_in.time:.2f}")

        # Get the init state
        body_q = state_in.body_q.clone()
        body_qd = state_in.body_qd.clone()
        body_f = state_in.body_f.clone()

        # Save the contact points to the state_in
        state_in.contact_count = model.contact_count
        state_in.contact_body = model.contact_body.clone()
        state_in.contact_point = model.contact_point.clone()
        state_in.contact_normal = model.contact_normal.clone()
        state_in.contact_point_idx = model.contact_point_idx.clone()

        # ======================================== START: CONTROL ========================================
        debug.section("CONTROL")
        body_f = body_f + forces_from_joint_actions(body_q, model.joint_parent,
                                            model.joint_child, model.joint_X_p,
                                            model.joint_X_c, model.joint_axis,
                                            model.joint_axis_mode,
                                            control.joint_act)
        state_in.body_f = body_f
        # ======================================== END: CONTROL ========================================

        # ======================================== START: INTEGRATION ========================================
        debug.section("INTEGRATION")
        for i in range(model.body_count):
            body_q[i], body_qd[i] = integrate_body(body_q[i], body_qd[i],
                                                   body_f[i],
                                                   model.body_inv_mass[i],
                                                   model.body_inv_inertia[i],
                                                   model.gravity, dt)
            debug.print(f"- Body {i}:")
            debug.indent()
            debug.print("Position:", body_q[i][:3])
            debug.print("Rotation:", quat_to_rotvec(body_q[i][3:]))
            debug.print("Linear Velocity:", body_qd[i][3:])
            debug.print("Angular Velocity:", body_qd[i][:3])
            debug.undent()
        # ======================================== END: INTEGRATION ========================================

        # ======================================== START: POSITION SOLVE ========================================
        debug.section("POSITION SOLVE")
        n_lambda = torch.tensor([0.0], device=model.device)
        for i in range(self.iterations):
            debug.subsection(f"ITERATION {i + 1}")

            # ----------------------------------- START: CONTACT CORRECTION -----------------------------------
            debug.print("- CONTACT DELTAS:")
            debug.indent()
            debug.print("Contact Count:", model.contact_count)

            if model.contact_count > 0:
                contact_deltas, lambda_ = get_ground_contact_deltas(
                    body_q, model.contact_count, model.contact_body,
                    model.contact_point, model.body_inv_mass,
                    model.body_inv_inertia)

                # Apply the contact deltas
                num_corrections = 0
                for b in range(model.body_count):
                    body_deltas = contact_deltas[model.contact_body == b]
                    if len(body_deltas) == 0:
                        continue

                    non_zero_mask = torch.any(body_deltas != 0.0, dim=1)
                    num_body_corrections = non_zero_mask.sum().item()
                    num_corrections += num_body_corrections
                    if num_body_corrections == 0:
                        continue
                    delta_q = torch.sum(body_deltas,
                                        dim=0) / num_body_corrections

                    body_q[b, :3] = body_q[b, :3] + delta_q[:3]
                    body_q[b, 3:] = normalize_quat(body_q[b, 3:] + delta_q[3:])

                    debug.print(f"- Body {b}:")
                    debug.indent()
                    debug.print("Position Delta:", delta_q[:3])
                    debug.print("Rotation Delta:", delta_q[3:])
                    debug.undent()

                if num_corrections > 0:
                    n_lambda = n_lambda + lambda_ / num_corrections
            debug.undent()
            # ----------------------------------- END: CONTACT CORRECTION -----------------------------------

            # ----------------------------------- START: JOINT CORRECTION -----------------------------------
            debug.print("- JOINT DELTAS:")
            joint_deltas, n_corr = get_joint_deltas(
                body_q, model.joint_parent, model.joint_child, model.joint_X_p,
                model.joint_X_c, model.joint_axis, model.body_inv_mass,
                model.body_inv_inertia)

            # Apply the joint deltas
            debug.indent()
            for b in range(model.body_count):
                debug.print(f"- Body {b}:")
                debug.indent()
                debug.print("Position Delta:", joint_deltas[b][:3])
                debug.print("Rotation Delta:", joint_deltas[b][3:])
                debug.undent()
                if n_corr[b] > 0:
                    body_q[b, :3] = body_q[b, :3] + joint_deltas[b][:3] / n_corr[b]
                    body_q[b, 3:] = normalize_quat(body_q[b, 3:] +
                                                   joint_deltas[b][3:] /
                                                   n_corr[b])
            debug.undent()
            # ----------------------------------- END: JOINT CORRECTION -----------------------------------

        # ======================================== END: POSITION SOLVE ========================================

        # ======================================== START: VELOCITY UPDATE ========================================
        debug.section("VELOCITY UPDATE")
        for b in range(model.body_count):
            body_qd[b] = numerical_qd(body_q[b], state_in.body_q[b], dt)
            debug.print(f"- Body {b}:")
            debug.indent()
            debug.print("Linear Velocity:", body_qd[b][3:])
            debug.print("Angular Velocity:", body_qd[b][:3])
            debug.undent()
        # ======================================== END: VELOCITY UPDATE ========================================

        # ======================================== START: VELOCITY SOLVE ========================================
        debug.section("VELOCITY SOLVE")
        if model.contact_count > 0:
            # ----------------------------------- START: FRICTION CORRECTION -----------------------------------
            debug.print("- DYNAMIC FRICTION DELTAS:")
            debug.indent()
            dynamic_friction_deltas = get_dynamic_friction_deltas(
                body_q, body_qd, model.contact_count, model.contact_body,
                model.contact_point, model.contact_normal, model.body_inv_mass,
                model.body_inv_inertia, model.dynamic_friction, n_lambda, dt)

            for b in range(model.body_count):
                body_deltas = dynamic_friction_deltas[model.contact_body == b]
                if len(body_deltas) == 0:
                    continue

                non_zero_mask = torch.any(body_deltas != 0.0, dim=1)
                num_body_corrections = non_zero_mask.sum().item()
                if num_body_corrections != 0:
                    delta_qd = torch.sum(body_deltas,
                                         dim=0) / num_body_corrections
                    body_qd[b] = body_qd[b] + delta_qd

                    debug.print(f"- Body {b}:")
                    debug.indent()
                    debug.print("Linear Velocity Delta:", delta_qd[3:])
                    debug.print("Angular Velocity Delta:", delta_qd[:3])
                    debug.undent()
            debug.undent()
            # ----------------------------------- END: FRICTION CORRECTION -----------------------------------

            # ----------------------------------- START: RESTITUTION CORRECTION -----------------------------------
            debug.print("- RESTITUTION DELTAS:")
            debug.indent()
            restitution_deltas = get_restitution_deltas(
                body_q, body_qd, state_in.body_qd, model.contact_count,
                model.contact_body, model.contact_point, model.contact_normal,
                model.body_inv_mass, model.body_inv_inertia, model.restitution)

            for b in range(model.body_count):
                body_deltas = restitution_deltas[model.contact_body == b]
                if len(body_deltas) == 0:
                    continue

                non_zero_mask = torch.any(body_deltas != 0.0, dim=1)
                num_body_corrections = non_zero_mask.sum().item()
                if num_body_corrections != 0:
                    delta_qd = torch.sum(body_deltas,
                                         dim=0) / num_body_corrections
                    body_qd[b] = body_qd[b] + delta_qd

                    debug.print(f"- Body {b}:")
                    debug.indent()
                    debug.print("Linear Velocity Delta:", delta_qd[3:])
                    debug.print("Angular Velocity Delta:", delta_qd[:3])
                    debug.undent()
            debug.undent()
            # ----------------------------------- END: RESTITUTION CORRECTION -----------------------------------

        # ======================================== END: VELOCITY SOLVE ========================================

        # Save the final state
        state_out.body_q = body_q
        state_out.body_qd = body_qd
        state_out.time = state_in.time + dt
