from typing import Tuple

import torch
from pbd_torch.constants import TRANSFORM_IDENTITY
from pbd_torch.correction import positional_deltas
from pbd_torch.correction import velocity_deltas
from pbd_torch.correction import joint_rot_deltas
from pbd_torch.integrator import SemiImplicitEulerIntegrator
from pbd_torch.model import Control
from pbd_torch.model import Model
from pbd_torch.model import State
from pbd_torch.transform import normalize_quat_batch
from pbd_torch.transform import quat_inv_batch
from pbd_torch.transform import quat_mul_batch
from pbd_torch.transform import transform_points_batch
from pbd_torch.utils import forces_from_joint_acts, compute_joint_coordinates


def numerical_qd(
    body_q: torch.Tensor,  # [body_count, 7, 1]
    body_q_prev: torch.Tensor,  # [body_count, 7, 1]
    h: float,
) -> torch.Tensor:
    x, q = body_q[:, :3], body_q[:, 3:]
    x_prev, q_prev = body_q_prev[:, :3], body_q_prev[:, 3:]

    # Compute the linear velocity
    v = (x - x_prev) / h  # [body_count, 3, 1]

    # Compute the angular velocity
    q_rel = quat_mul_batch(q, quat_inv_batch(q_prev))  # [body_count, 4, 1]
    w = 2 * q_rel[:, 1:] / h  # [body_count, 3, 1]

    # Flip the omega where the scalar part is negative
    negative_mask = (q_rel[:, 0, 0] < 0).flatten()  # [body_count]
    w[negative_mask] = -w[negative_mask]  # [body_count, 3, 1]

    qd = torch.cat([w, v], dim=1)  # [body_count, 6, 1]
    return qd

def normalize_values(
        values: torch.Tensor,
        normalization_factor: torch.Tensor
) -> torch.Tensor:
    epsilon = torch.tensor(1e-6, device=values.device)
    return values / (normalization_factor + epsilon)

def gap_function(p_a: torch.Tensor, p_b: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    contact_vector = p_b - p_a  # Vector from body A to body B [N, 3, 1]
    gap = torch.matmul(contact_vector.transpose(2, 1), n).flatten()  # [N]
    return gap

def get_ground_contact_deltas(
    body_q: torch.Tensor,
    contact_count: int,
    contact_body: torch.Tensor,
    contact_point: torch.Tensor,
    contact_normal: torch.Tensor,
    contact_point_ground: torch.Tensor,
    body_inv_mass: torch.Tensor,
    body_inv_inertia: torch.Tensor,
    lambda_: torch.Tensor,
    compliance: torch.Tensor,
    h: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = body_q.device
    body_count = body_q.shape[0]

    # Initialize outputs
    body_deltas = torch.zeros((body_count, 7, 1), device=device)
    dlambda_ = torch.zeros_like(lambda_, device=device)  # [body_count, 1, 1]

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
    penetration_depth = torch.matmul(contact_vector.transpose(2, 1), -contact_normal).view(-1)  # [P]

    penetration_mask = penetration_depth < 0
    corrected_bodies = contact_body[penetration_mask].view(-1)

    # Compute position corrections for valid contacts
    dbody_q_batch, _, dlambda_batch = positional_deltas(
        body_trans_a=body_q_a[penetration_mask].view(-1, 7, 1),
        body_trans_b=body_q_b[penetration_mask].view(-1, 7, 1),
        r_a=contact_point[penetration_mask].view(-1, 3, 1),
        r_b=p_b[penetration_mask].view(-1, 3, 1),
        m_a_inv=m_inv_a[penetration_mask].view(-1, 1, 1),
        m_b_inv=m_inv_b[penetration_mask].view(-1, 1, 1),
        I_a_inv=I_inv_a[penetration_mask].view(-1, 3, 3),
        I_b_inv=I_inv_b[penetration_mask].view(-1, 3, 3),
        lambda_=lambda_[contact_body][penetration_mask].view(-1, 1, 1),
        compliance=compliance[contact_body][penetration_mask].view(-1, 1, 1),
        h=h,
    )

    # Compute, how many penetrations we have per body
    num_corrections = torch.bincount(corrected_bodies, minlength=body_count)  # [body_count]

    # Add the deltas to the body_deltas and lambda_n
    body_deltas = body_deltas.scatter_add(
        0, corrected_bodies.view(-1, 1, 1).expand_as(dbody_q_batch), dbody_q_batch
    )
    dlambda_ = dlambda_.scatter_add(0, corrected_bodies.view(-1, 1, 1), dlambda_batch)

    # Normalize the deltas by the number of corrections
    body_deltas = normalize_values(body_deltas, num_corrections.view(-1, 1, 1))
    lambda_ = lambda_ + normalize_values(dlambda_, num_corrections.view(-1, 1, 1))

    return body_deltas, lambda_

def get_joint_pos_deltas(
    body_q: torch.Tensor,
    joint_parent: torch.Tensor,
    joint_child: torch.Tensor,
    joint_X_p: torch.Tensor,
    joint_X_c: torch.Tensor,
    body_inv_mass: torch.Tensor,
    body_inv_inertia: torch.Tensor,
    lambda_: torch.Tensor,
    compliance: torch.Tensor,
    h: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    body_count = body_q.shape[0]

    dbody_q_p_batch, dbody_q_c_batch, dlambda = positional_deltas(
        body_trans_a=body_q[joint_parent].view(-1, 7, 1),
        body_trans_b=body_q[joint_child].view(-1, 7, 1),
        r_a=joint_X_p[:, :3].view(-1, 3, 1),
        r_b=joint_X_c[:, :3].view(-1, 3, 1),
        m_a_inv=body_inv_mass[joint_parent].view(-1, 1, 1),
        m_b_inv=body_inv_mass[joint_child].view(-1, 1, 1),
        I_a_inv=body_inv_inertia[joint_parent].view(-1, 3, 3),
        I_b_inv=body_inv_inertia[joint_child].view(-1, 3, 3),
        lambda_=lambda_.view(-1, 1, 1),
        compliance=compliance.view(-1, 1, 1),
        h=h
    )

    lambda_ = lambda_ + dlambda

    # Compute correction masks
    correction_mask_p = torch.any(dbody_q_p_batch != 0.0, dim=1).flatten()
    correction_mask_c = torch.any(dbody_q_c_batch != 0.0, dim=1).flatten()

    # Initialize joint_deltas and num_corrections
    body_deltas = torch.zeros((body_count, 7, 1), device=body_q.device)
    num_corrections = torch.zeros(body_count, device=body_q.device, dtype=torch.int64)

    # Compute parent corrections
    parent_indices = joint_parent[correction_mask_p]
    parent_deltas = dbody_q_p_batch[correction_mask_p]
    parent_counts = torch.ones_like(
        parent_indices, device=body_q.device, dtype=torch.int64
    )

    # Use scatter_add_ for parent bodies
    body_deltas.scatter_add_(
        0, parent_indices.view(-1, 1, 1).expand_as(parent_deltas), parent_deltas
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
        0, child_indices.view(-1, 1, 1).expand_as(child_deltas), child_deltas
    )
    num_corrections.scatter_add_(0, child_indices, child_counts)

    body_deltas = normalize_values(body_deltas, num_corrections.view(-1, 1, 1))

    return body_deltas, lambda_

def get_joint_rot_deltas(
        body_q: torch.Tensor,
        joint_parent: torch.Tensor,
        joint_child: torch.Tensor,
        joint_X_p: torch.Tensor,
        joint_X_c: torch.Tensor,
        joint_axis: torch.Tensor,
        body_inv_inertia: torch.Tensor,
        lambda_: torch.Tensor,
        compliance: torch.Tensor,
        h: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    body_count = body_q.shape[0]

    dbody_q_p_batch, dbody_q_c_batch, dlambda = joint_rot_deltas(
        body_q_p=body_q[joint_parent].view(-1, 7, 1),
        body_q_c=body_q[joint_child].view(-1, 7, 1),
        X_p=joint_X_p.view(-1, 7, 1),
        X_c=joint_X_c.view(-1, 7, 1),
        joint_axis=joint_axis.view(-1, 3, 1),
        I_p_inv=body_inv_inertia[joint_parent].view(-1, 3, 3),
        I_c_inv=body_inv_inertia[joint_child].view(-1, 3, 3),
        lambda_=lambda_.view(-1, 1, 1),
        compliance=compliance.view(-1, 1, 1),
        h=h
    )

    lambda_ = lambda_ + dlambda

    # Compute correction masks
    correction_mask_p = torch.any(dbody_q_p_batch != 0.0, dim=1).flatten()
    correction_mask_c = torch.any(dbody_q_c_batch != 0.0, dim=1).flatten()

    # Initialize joint_deltas and num_corrections
    body_deltas = torch.zeros((body_count, 7, 1), device=body_q.device)
    num_corrections = torch.zeros(body_count, device=body_q.device, dtype=torch.int64)

    # Compute parent corrections
    parent_indices = joint_parent[correction_mask_p]
    parent_deltas = dbody_q_p_batch[correction_mask_p]
    parent_counts = torch.ones_like(
        parent_indices, device=body_q.device, dtype=torch.int64
    )

    # Use scatter_add_ for parent bodies
    body_deltas.scatter_add_(
        0, parent_indices.view(-1, 1, 1).expand_as(parent_deltas), parent_deltas
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
        0, child_indices.view(-1, 1, 1).expand_as(child_deltas), child_deltas
    )
    num_corrections.scatter_add_(0, child_indices, child_counts)

    body_deltas = normalize_values(body_deltas, num_corrections.view(-1, 1, 1))

    return body_deltas, lambda_

def get_velocity_deltas(
    body_q: torch.Tensor,
    body_qd: torch.Tensor,
    body_qd_prev: torch.Tensor,
    contact_body: torch.Tensor,
    contact_point: torch.Tensor,
    contact_normal: torch.Tensor,
    contact_point_ground: torch.Tensor,
    body_inv_mass: torch.Tensor,
    body_inv_inertia: torch.Tensor,
    restitution: torch.Tensor,
    dynamic_friction: torch.Tensor,
    lambda_n: torch.Tensor,
    h: float
) -> torch.Tensor:
    device = body_q.device
    body_count = body_q.shape[0]

    # Initialize output
    body_deltas = torch.zeros((body_count, 6, 1), device=device)

    # Get body states for all contacts
    contact_body_q = body_q[contact_body]  # [C, 7, 1]
    contact_body_qd = body_qd[contact_body]  # [C, 6, 1]
    contact_body_qd_prev = body_qd_prev[contact_body]  # [C, 6, 1]

    # Get contact points in world space
    p_a = transform_points_batch(contact_point, contact_body_q)  # [C, 3, 1]
    p_b = contact_point_ground  # [C, 3, 1]

    # Create mask for valid contacts (where contact_point is below the threshold)
    gap = gap_function(p_a, p_b, contact_normal)  # [C, 1]
    valid_mask = (gap < 0.0).view(-1)  # [C]
    valid_mask = torch.ones(valid_mask.shape, device=device, dtype=torch.bool)  # [C]
    valid_body_idxs = contact_body[valid_mask]  # [valid_contact_count]

    # Apply batched restitution calculation
    dbody_qd = velocity_deltas(
        body_q=contact_body_q[valid_mask].view(-1, 7, 1),
        body_qd=contact_body_qd[valid_mask].view(-1, 6, 1),
        body_qd_prev=contact_body_qd_prev[valid_mask].view(-1, 6, 1),
        r=contact_point[valid_mask].view(-1, 3, 1),
        n=contact_normal[valid_mask].view(-1, 3, 1),
        m_inv=body_inv_mass[valid_body_idxs].view(-1, 1, 1),
        I_inv=body_inv_inertia[valid_body_idxs].view(-1, 3, 3),
        restitution=restitution[valid_body_idxs].view(-1, 1, 1),
        dynamic_friction=dynamic_friction[valid_body_idxs].view(-1, 1, 1),
        lambda_n=lambda_n[valid_body_idxs].view(-1, 1, 1),
        h=h,
    )

    # Compute how many corrections we have per body
    num_corrections = torch.bincount(valid_body_idxs, minlength=body_count)  # [body_count]

    # Add the deltas to the body_deltas
    body_deltas = body_deltas.scatter_add(
        0, valid_body_idxs.view(-1, 1, 1).expand_as(dbody_qd), dbody_qd
    )

    # Normalize the deltas by the number of corrections
    body_deltas = normalize_values(body_deltas, num_corrections.view(-1, 1, 1))

    return body_deltas


class XPBDEngine:

    def __init__(self, model: Model,
                 pos_iters: int = 5,
                 device: torch.device = torch.device("cpu"),
                 contact_compliance: float = 1e-6,
                 joint_compliance: float = 1e-4):
        self.pos_iters = pos_iters
        self.integrator = SemiImplicitEulerIntegrator(use_local_omega=False, device=device)
        self.model = model

        self.contact_compliance = contact_compliance
        self.joint_compliance = joint_compliance

    def simulate(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        dt: float,
    ):
        # Get the init state
        body_q = state_in.body_q.clone()
        body_qd = state_in.body_qd.clone()
        body_f = state_in.body_f.clone()

        body_f = body_f + forces_from_joint_acts(
            control.joint_act,
            state_in.joint_q,
            state_in.joint_qd,
            self.model.joint_ke,
            self.model.joint_kd,
            body_q,
            self.model.joint_parent,
            self.model.joint_child,
            self.model.joint_X_p,
            self.model.joint_X_c,
        )
        state_in.body_f = body_f

        body_q, body_qd = self.integrator.integrate(
            body_q,
            body_qd,
            body_f,
            self.model.body_inv_mass,
            self.model.body_inv_inertia,
            self.model.g_accel,
            dt,
        )

        body_q, lambda_n = self.position_solve_jacobi(
            body_q,
            state_in.contact_count,
            state_in.contact_body_indices_flat,
            state_in.contact_points_flat,
            state_in.contact_normals_flat,
            state_in.contact_points_ground_flat,
            dt
        )

        body_qd = numerical_qd(body_q, state_in.body_q, dt)

        velocity_deltas = get_velocity_deltas(
            body_q,
            body_qd,
            state_in.body_qd,
            state_in.contact_body_indices_flat,
            state_in.contact_points_flat,
            state_in.contact_normals_flat,
            state_in.contact_points_ground_flat,
            self.model.body_inv_mass,
            self.model.body_inv_inertia,
            self.model.restitution,
            self.model.dynamic_friction,
            lambda_n,
            dt,
        )
        body_qd = body_qd + velocity_deltas

        joint_q, joint_qd = compute_joint_coordinates(body_q,
                                                      body_qd,
                                                      self.model.joint_parent,
                                                      self.model.joint_child,
                                                      self.model.joint_X_p,
                                                      self.model.joint_X_c,
                                                      self.model.joint_axis)
        # Save the final state
        state_out.body_q = body_q
        state_out.body_qd = body_qd
        state_out.joint_q = joint_q
        state_out.joint_qd = joint_qd
        state_out.time = state_in.time + dt

    def position_solve_jacobi(self,
        body_q: torch.Tensor,
        contact_count: int,
        contact_body_indices_flat: torch.Tensor,
        contact_points_flat: torch.Tensor,
        contact_normals_flat: torch.Tensor,
        contact_points_ground_flat: torch.Tensor,
        dt: float
    ) :
        lambda_n = torch.zeros((self.model.body_count, 1, 1), device=body_q.device)
        lambda_j_pos = torch.zeros((self.model.joint_count, 1, 1), device=body_q.device)
        lambda_j_rot = torch.zeros((self.model.joint_count, 1, 1), device=body_q.device)
        contact_compliance = torch.full((self.model.body_count, 1, 1), self.contact_compliance, device=body_q.device)
        joint_compliance = torch.full((self.model.joint_count, 1, 1), self.joint_compliance, device=body_q.device)

        for _ in range(self.pos_iters):
           # ----------------------------------- START: CONTACT CORRECTION -----------------------------------
            dbody_q_n, lambda_n = get_ground_contact_deltas(
                body_q,
                contact_count,
                contact_body_indices_flat,
                contact_points_flat,
                contact_normals_flat,
                contact_points_ground_flat,
                self.model.body_inv_mass,
                self.model.body_inv_inertia,
                lambda_n,
                contact_compliance,
                dt
            )

            dbody_q_j_pos, lambda_j_pos = get_joint_pos_deltas(
                body_q,
                self.model.joint_parent,
                self.model.joint_child,
                self.model.joint_X_p,
                self.model.joint_X_c,
                self.model.body_inv_mass,
                self.model.body_inv_inertia,
                lambda_j_pos,
                joint_compliance,
                dt,
            )

            dbody_q_j_rot, lambda_j_rot = get_joint_rot_deltas(
                body_q,
                self.model.joint_parent,
                self.model.joint_child,
                self.model.joint_X_p,
                self.model.joint_X_c,
                self.model.joint_axis,
                self.model.body_inv_inertia,
                lambda_j_rot,
                joint_compliance,
                dt,
            )

            body_q = body_q + dbody_q_n + dbody_q_j_pos + dbody_q_j_rot
            body_q[:, 3:] = normalize_quat_batch(body_q[:, 3:].clone())

        return body_q, lambda_n