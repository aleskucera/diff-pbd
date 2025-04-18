import os
import time
from typing import Tuple

import torch
from pbd_torch.constants import TRANSFORM_IDENTITY
from pbd_torch.correction import ground_dynamic_friction_delta_batch
from pbd_torch.correction import ground_restitution_delta_batch
from pbd_torch.correction import joint_deltas_batch
from pbd_torch.correction import positional_deltas_batch
from pbd_torch.integrator import SemiImplicitEulerIntegrator
from pbd_torch.logger import DebugLogger
from pbd_torch.model import Control
from pbd_torch.model import JOINT_MODE_FORCE
from pbd_torch.model import JOINT_MODE_TARGET_POSITION
from pbd_torch.model import JOINT_MODE_TARGET_VELOCITY
from pbd_torch.model import Model
from pbd_torch.model import State
from pbd_torch.transform import normalize_quat_batch
from pbd_torch.transform import quat_inv_batch
from pbd_torch.transform import quat_mul_batch
from pbd_torch.transform import rotate_vectors_batch
from pbd_torch.transform import rotate_vectors_inverse_batch
from pbd_torch.transform import transform_multiply_batch
from pbd_torch.transform import transform_points_batch
from pbd_torch.utils import forces_from_joint_actions

os.environ["DEBUG"] = "false"

def numerical_qd(
    body_q: torch.Tensor,  # [body_count, 7]
    body_q_prev: torch.Tensor,  # [body_count, 7]
    dt: float,  # float
) -> torch.Tensor:
    """Computes numerical velocities from positions at current and previous time steps.

    Args:
        body_q (torch.Tensor): Current body state
        body_q_prev (torch.Tensor): Previous body state
        dt (float): Time step

    Returns:
        torch.Tensor: Computed velocities
    """

    x, q = body_q[:, :3], body_q[:, 3:]
    x_prev, q_prev = body_q_prev[:, :3], body_q_prev[:, 3:]

    # Compute the linear velocity
    v = (x - x_prev) / dt  # [body_count, 3]

    # Compute the angular velocity
    q_rel = quat_mul_batch(quat_inv_batch(q_prev), q)  # [body_count, 4]
    w = 2 * q_rel[:, 1:] / dt  # [body_count, 3]

    # Flip the omega where the scalar part is negative
    negative_mask = q_rel[:, 0] < 0  # [body_count]
    w[negative_mask] = -w[negative_mask]  # [body_count, 3]

    qd = torch.cat([w, v], dim=1)  # [body_count, 6]
    return qd


def get_ground_contact_deltas(
    body_q: torch.Tensor,
    contact_count: int,
    contact_body: torch.Tensor,
    contact_point: torch.Tensor,
    contact_normal: torch.Tensor,
    contact_point_ground: torch.Tensor,
    body_inv_mass: torch.Tensor,
    body_inv_inertia: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes position corrections for ground contacts.

    In contacts with ground we assume that the ground is a static body with infinite mass, and
    infinite inertia. This simplifies the computation of the contact correction.

    Here we compute the position correction and the Lagrange multiplier for the normal force.
    """
    device = body_q.device
    body_count = body_q.shape[0]

    # Initialize outputs
    body_deltas = torch.zeros((body_count, 7, 1), device=device)
    lambda_n = torch.zeros((body_count, 1), device=device)

    # Get body states for all contacts at once
    body_q_a = body_q[contact_body]  # [contact_count, 7, 1]
    m_inv_a = body_inv_mass[contact_body]  # [contact_count, 1, 1]
    I_inv_a = body_inv_inertia[contact_body]  # [contact_count, 3, 3]
    p_a = transform_points_batch(contact_point, body_q_a)  # [contact_count, 3, 1]

    transform_identity = TRANSFORM_IDENTITY.to(device).unsqueeze(0) # [1, 7]
    body_q_b = transform_identity.repeat(contact_count, 1).unsqueeze(2)  # [contact_count, 7, 1]
    m_inv_b = torch.zeros((contact_count, 1, 1), device=device)  # [contact_count, 1, 1]
    I_inv_b = torch.zeros((contact_count, 3, 3), device=device)  # [contact_count, 3, 3]
    p_b = contact_point_ground  # [contact_count, 3, 1]

    # Check the penetration depth in the normal direction
    # Using dot product of (point_b - point_a) with normal
    # If dot product is negative, there's penetration
    contact_vector = p_b - p_a  # Vector from ground to contact [contact_count, 3, 1]
    penetration_depth = torch.matmul(contact_vector.transpose(2, 1), contact_normal).view(-1)  # [P, 1]

    penetration_mask = penetration_depth < 0
    corrected_bodies = contact_body[penetration_mask].view(-1)

    # Compute position corrections for valid contacts
    dbody_q_batch, _, d_lambda_batch = positional_deltas_batch(
        body_trans_a=body_q_a[penetration_mask].view(-1, 7),
        body_trans_b=body_q_b[penetration_mask].view(-1, 7),
        r_a=contact_point[penetration_mask].view(-1, 3),
        r_b=p_b[penetration_mask].view(-1, 3),
        m_a_inv=m_inv_a[penetration_mask].view(-1, 1),
        m_b_inv=m_inv_b[penetration_mask].view(-1, 1),
        I_a_inv=I_inv_a[penetration_mask].view(-1, 3, 3),
        I_b_inv=I_inv_b[penetration_mask].view(-1, 3, 3),
    )

    # Compute, how many penetrations we have per body
    num_corrections = torch.bincount(
        corrected_bodies, minlength=body_count
    )  # [body_count]

    # Add the deltas to the body_deltas and lambda_n
    body_deltas = body_deltas.scatter_add(
        0, corrected_bodies.unsqueeze(1).expand_as(dbody_q_batch), dbody_q_batch
    )
    lambda_n = lambda_n.scatter_add(0, corrected_bodies, d_lambda_batch)

    # Normalize the deltas by the number of corrections
    body_deltas = normalize_values(body_deltas, num_corrections.unsqueeze(1))
    lambda_n = normalize_values(lambda_n, num_corrections)

    return body_deltas, lambda_n


def get_joint_deltas(
    body_q: torch.Tensor,
    joint_parent: torch.Tensor,
    joint_child: torch.Tensor,
    joint_X_p: torch.Tensor,
    joint_X_c: torch.Tensor,
    joint_axis: torch.Tensor,
    body_inv_mass: torch.Tensor,
    body_inv_inertia: torch.Tensor,
) -> torch.Tensor:
    """Computes position corrections for joint constraints.

    Args:
        body_q (torch.Tensor): Body states [body_count, 7]
        joint_parent (torch.Tensor): Parent body indices [joint_count]
        joint_child (torch.Tensor): Child body indices [joint_count]
        joint_X_p (torch.Tensor): Parent joint transforms [joint_count, 7]
        joint_X_c (torch.Tensor): Child joint transforms [joint_count, 7]
        joint_axis (torch.Tensor): Joint axes [joint_count, 3]
        body_inv_mass (torch.Tensor): Inverse masses [body_count]
        body_inv_inertia (torch.Tensor): Inverse inertia tensors [body_count]

    Returns:
        tuple: (joint_deltas, num_corrections)
    """
    body_count = body_q.shape[0]

    dbody_q_p_batch, dbody_q_c_batch = joint_deltas_batch(
        body_q_p=body_q[joint_parent].view(-1, 7),
        body_q_c=body_q[joint_child].view(-1, 7),
        X_p=joint_X_p.view(-1, 7),
        X_c=joint_X_c.view(-1, 7),
        joint_axis=joint_axis.view(-1, 3),
        m_p_inv=body_inv_mass[joint_parent].view(-1, 1),
        m_c_inv=body_inv_mass[joint_child].view(-1, 1),
        I_p_inv=body_inv_inertia[joint_parent].view(-1, 3, 3),
        I_c_inv=body_inv_inertia[joint_child].view(-1, 3, 3),
    )

    # Compute correction masks
    correction_mask_p = torch.any(dbody_q_p_batch != 0.0, dim=1)
    correction_mask_c = torch.any(dbody_q_c_batch != 0.0, dim=1)

    # Initialize joint_deltas and num_corrections
    body_deltas = torch.zeros((body_count, 7), device=body_q.device)
    num_corrections = torch.zeros(body_count, device=body_q.device, dtype=torch.int64)

    # Compute parent corrections
    parent_indices = joint_parent[correction_mask_p]
    parent_deltas = dbody_q_p_batch[correction_mask_p]
    parent_counts = torch.ones_like(
        parent_indices, device=body_q.device, dtype=torch.int64
    )

    # Use scatter_add_ for parent bodies
    body_deltas.scatter_add_(
        0, parent_indices.unsqueeze(-1).expand_as(parent_deltas), parent_deltas
    )
    num_corrections.scatter_add_(0, parent_indices, parent_counts)

    # Compute child corrections
    child_indices = joint_child[correction_mask_c]
    child_deltas = dbody_q_c_batch[correction_mask_c]
    child_counts = torch.ones_like(
        child_indices, device=body_q.device, dtype=torch.int64
    )

    # Use scatter_add_ for child bodies
    body_deltas.scatter_add_(
        0, child_indices.unsqueeze(-1).expand_as(child_deltas), child_deltas
    )
    num_corrections.scatter_add_(0, child_indices, child_counts)

    body_deltas = normalize_values(body_deltas, num_corrections.unsqueeze(1))

    return body_deltas


def gap_function(p_a: torch.Tensor, p_b: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Computes penetration depths for a set of contact points.
    Args:
        p_a (torch.Tensor): Contact points in body A [N, 3]
        p_b (torch.Tensor): Contact points in body B [N, 3]
        n (torch.Tensor): Contact normals, pointing from A towards B [N, 3]
    Returns:
        torch.Tensor: Computed penetration depths [N]
    """
    contact_vector = p_b - p_a  # Vector from body A to body B [N, 3]
    gap = torch.bmm(
        contact_vector.unsqueeze(1), n.unsqueeze(2)
    ).squeeze()  # Dot product with normal [N]
    return gap


def normalize_values(
    values: torch.Tensor, normalization_factor: torch.Tensor
) -> torch.Tensor:
    epsilon = torch.tensor(1e-6, device=values.device)
    return values / (normalization_factor + epsilon)


def get_dynamic_friction_deltas(
    body_q: torch.Tensor,
    body_qd: torch.Tensor,
    contact_count: int,
    contact_body: torch.Tensor,
    contact_point: torch.Tensor,
    contact_normal: torch.Tensor,
    contact_point_ground: torch.Tensor,
    body_inv_mass: torch.Tensor,
    body_inv_inertia: torch.Tensor,
    dynamic_friction: torch.Tensor,
    lambda_n: torch.Tensor,
    gap_threshold: float,
    dt: float,
) -> torch.Tensor:
    """Computes velocity corrections for dynamic friction using batch operations."""
    device = body_q.device
    body_count = body_q.shape[0]

    # Initialize output
    body_deltas = torch.zeros((body_count, 6), device=device)

    # Get body states for all contacts
    contact_body_q = body_q[contact_body]  # [contact_count, 7]
    contact_body_qd = body_qd[contact_body]  # [contact_count, 6]

    # Get the contact points in world space
    p_a = transform_points_batch(contact_point, contact_body_q)  # [contact_count, 3]
    p_b = contact_point_ground  # [contact_count, 3]

    # Create mask for valid contacts (where contact_point is below the threshold)
    gap = gap_function(p_a, p_b, contact_normal)  # [contact_count]
    valid_mask = (gap < gap_threshold).view(-1)  # [contact_count]
    valid_body_idxs = contact_body[valid_mask]  # [valid_contact_count]

    # Apply batched dynamic friction calculation
    dbody_qd_batch = ground_dynamic_friction_delta_batch(
        body_q=contact_body_q[valid_mask].view(-1, 7),
        body_qd=contact_body_qd[valid_mask].view(-1, 6),
        r=contact_point[valid_mask].view(-1, 3),
        n=contact_normal[valid_mask].view(-1, 3),
        m_inv=body_inv_mass[valid_body_idxs].view(-1, 1),
        I_inv=body_inv_inertia[valid_body_idxs].view(-1, 3, 3),
        dynamic_friction=dynamic_friction[valid_body_idxs].view(-1),
        lambda_n=lambda_n[valid_body_idxs].view(-1),
        dt=dt,
    )

    # Compute how many corrections we have per body
    num_corrections = torch.bincount(
        valid_body_idxs, minlength=body_count
    )  # [body_count]

    # Add the deltas to the body_deltas
    body_deltas = body_deltas.scatter_add(
        0, valid_body_idxs.unsqueeze(1).expand_as(dbody_qd_batch), dbody_qd_batch
    )

    # Normalize the deltas by the number of corrections
    body_deltas = normalize_values(body_deltas, num_corrections.unsqueeze(1))

    return body_deltas


def get_restitution_deltas(
    body_q: torch.Tensor,
    body_qd: torch.Tensor,
    body_qd_prev: torch.Tensor,
    contact_count: int,
    contact_body: torch.Tensor,
    contact_point: torch.Tensor,
    contact_normal: torch.Tensor,
    contact_point_ground: torch.Tensor,
    body_inv_mass: torch.Tensor,
    body_inv_inertia: torch.Tensor,
    restitution: torch.Tensor,
) -> torch.Tensor:
    """Computes velocity corrections for restitution handling using batch operations."""
    device = body_q.device
    body_count = body_q.shape[0]

    # Initialize output
    body_deltas = torch.zeros((body_count, 6), device=device)

    # Get body states for all contacts
    contact_body_q = body_q[contact_body]  # [contact_count, 7]
    contact_body_qd = body_qd[contact_body]  # [contact_count, 6]
    contact_body_qd_prev = body_qd_prev[contact_body]  # [contact_count, 6]

    # Get contact points in world space
    p_a = transform_points_batch(contact_point, contact_body_q)  # [contact_count, 3]
    p_b = contact_point_ground  # [contact_count, 3]

    # Create mask for valid contacts (where contact_point is below the threshold)
    gap = gap_function(p_a, p_b, contact_normal)  # [contact_count]
    valid_mask = (gap < 0.0).view(-1)  # [contact_count]
    valid_body_idxs = contact_body[valid_mask]  # [valid_contact_count]

    # Apply batched restitution calculation
    dbody_qd_batch = ground_restitution_delta_batch(
        body_q=contact_body_q[valid_mask].view(-1, 7),
        body_qd=contact_body_qd[valid_mask].view(-1, 6),
        body_qd_prev=contact_body_qd_prev[valid_mask].view(-1, 6),
        r=contact_point[valid_mask].view(-1, 3),
        n=contact_normal[valid_mask].view(-1, 3),
        m_inv=body_inv_mass[valid_body_idxs].view(-1, 1),
        I_inv=body_inv_inertia[valid_body_idxs].view(-1, 3, 3),
        restitution=restitution[valid_body_idxs].view(-1),
    )

    # Compute how many corrections we have per body
    num_corrections = torch.bincount(
        valid_body_idxs, minlength=body_count
    )  # [body_count]

    # Add the deltas to the body_deltas
    body_deltas = body_deltas.scatter_add(
        0, valid_body_idxs.unsqueeze(1).expand_as(dbody_qd_batch), dbody_qd_batch
    )

    # Normalize the deltas by the number of corrections
    body_deltas = normalize_values(body_deltas, num_corrections.unsqueeze(1))

    return body_deltas


class XPBDEngine:

    def __init__(self, model: Model, iterations: int = 2, device: torch.device = torch.device("cpu")):
        self.iterations = iterations
        self.integrator = SemiImplicitEulerIntegrator(
            use_local_omega=True, device=device
        )
        self.model = model

        self.logger = DebugLogger()

    def simulate(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        dt: float,
    ):
        self.logger.section(f"TIME: {state_in.time:.2f}")

        # Get the init state
        body_q = state_in.body_q.clone()
        body_qd = state_in.body_qd.clone()
        body_f = state_in.body_f.clone()

        # ======================================== START: CONTROL ========================================
        # self.logger.section("CONTROL")
        control_time = time.time()
        body_f = body_f + forces_from_joint_actions(
            body_q,
            self.model.joint_parent,
            self.model.joint_child,
            self.model.joint_X_p,
            self.model.joint_X_c,
            control.joint_act,
        )
        # TODO: Transform body_f
        state_in.body_f = body_f
        self.logger.print(f"Control time: {time.time() - control_time:.5f}")
        # ======================================== END: CONTROL ========================================

        # ======================================== START: INTEGRATION ========================================
        int_time = time.time()
        body_q, body_qd = self.integrator.integrate(
            body_q,
            body_qd,
            body_f,
            self.model.body_inv_mass,
            self.model.body_inv_inertia,
            self.model.g_accel,
            dt,
        )
        self.logger.print(f"Integration time: {time.time() - int_time:.5f}")

        # ======================================== END: INTEGRATION ========================================

        # ======================================== START: POSITION SOLVE ========================================
        # self.logger.section("POSITION SOLVE")
        position_solve_time = time.time()
        n_lambda = torch.zeros(self.model.body_count, device=body_q.device)
        for _ in range(self.iterations):

            # self.logger.subsection(f"ITERATION {i + 1}")
            #
            # # ----------------------------------- START: CONTACT CORRECTION -----------------------------------
            # self.logger.print("- CONTACT DELTAS:")
            # self.logger.indent()
            # self.logger.print("Contact Count:", model.contact_count)

            contact_corr_time = time.time()
            contact_body_q_deltas, lambda_n_deltas = get_ground_contact_deltas(
                body_q,
                state_in.contact_count,
                state_in.contact_body_indices_flat,
                state_in.contact_points_flat,
                state_in.contact_normals_flat,
                state_in.contact_points_ground_flat,
                self.model.body_inv_mass,
                self.model.body_inv_inertia,
            )

            body_q = body_q + contact_body_q_deltas
            body_q[:, 3:] = normalize_quat_batch(body_q[:, 3:])

            n_lambda = n_lambda + lambda_n_deltas

            self.logger.print(
                f"Contact correction time: {time.time() - contact_corr_time:.5f}"
            )

            # self.logger.undent()
            # ----------------------------------- END: CONTACT CORRECTION -----------------------------------

            # ----------------------------------- START: JOINT CORRECTION -----------------------------------
            # self.logger.print("- JOINT DELTAS:")
            joint_corr_time = time.time()
            joint_body_q_deltas = get_joint_deltas(
                body_q,
                self.model.joint_parent,
                self.model.joint_child,
                self.model.joint_X_p,
                self.model.joint_X_c,
                self.model.joint_axis,
                self.model.body_inv_mass,
                self.model.body_inv_inertia,
            )

            body_q = body_q + joint_body_q_deltas
            body_q[:, 3:] = normalize_quat_batch(body_q[:, 3:])

            self.logger.print(
                f"Joint correction time: {time.time() - joint_corr_time:.5f}"
            )

            # self.logger.undent()
            # ----------------------------------- END: JOINT CORRECTION -----------------------------------
        self.logger.print(
            f"Position solve time: {time.time() - position_solve_time:.5f}"
        )
        # ======================================== END: POSITION SOLVE ========================================

        # ======================================== START: VELOCITY UPDATE ========================================
        # self.logger.section("VELOCITY UPDATE")
        velocity_update_time = time.time()
        body_qd = numerical_qd(body_q, state_in.body_q, dt)
        self.logger.print(
            f"Velocity update time: {time.time() - velocity_update_time:.5f}"
        )
        # ======================================== END: VELOCITY UPDATE ========================================

        # ======================================== START: VELOCITY SOLVE ========================================
        velocity_solve_time = time.time()
        # ----------------------------------- START: FRICTION CORRECTION -----------------------------------
        dynamic_friction_deltas = get_dynamic_friction_deltas(
            body_q,
            body_qd,
            state_in.contact_count,
            state_in.contact_body_indices_flat,
            state_in.contact_points_flat,
            state_in.contact_normals_flat,
            state_in.contact_points_ground_flat,
            self.model.body_inv_mass,
            self.model.body_inv_inertia,
            self.model.dynamic_friction,
            n_lambda,
            self.model.dynamic_friction_threshold,
            dt,
        )

        body_qd = body_qd
        # ----------------------------------- END: FRICTION CORRECTION -----------------------------------

        # ----------------------------------- START: RESTITUTION CORRECTION -----------------------------------
        restitution_deltas = get_restitution_deltas(
            body_q,
            body_qd,
            state_in.body_qd,
            state_in.contact_count,
            state_in.contact_body_indices_flat,
            state_in.contact_points_flat,
            state_in.contact_normals_flat,
            state_in.contact_points_ground_flat,
            self.model.body_inv_mass,
            self.model.body_inv_inertia,
            self.model.restitution,
        )

        body_qd = body_qd + restitution_deltas + dynamic_friction_deltas
        # ----------------------------------- END: RESTITUTION CORRECTION -----------------------------------

        self.logger.print(
            f"Velocity solve time: {time.time() - velocity_solve_time:.5f}"
        )
        # ======================================== END: VELOCITY SOLVE ========================================

        # Save the final state
        state_out.body_q = body_q
        state_out.body_qd = body_qd
        state_out.time = state_in.time + dt
