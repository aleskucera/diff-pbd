from typing import Tuple

import torch
from pbd_torch.transform import rotate_vector
from pbd_torch.transform import rotate_vectors_batch
from pbd_torch.transform import transform_multiply_batch
from pbd_torch.transform import transform_points_batch


def skew_symmetric_matrix_batch(vectors: torch.Tensor) -> torch.Tensor:
    """
    Compute the skew-symmetric matrix of a batch of vectors.
    :param vectors: Tensor of shape [N, 3].
    :return: Tensor of shape [N, 3, 3] with skew-symmetric matrices.
    """
    N = vectors.shape[0]
    skew = torch.zeros(N, 3, 3, device=vectors.device, dtype=vectors.dtype)
    skew[:, 0, 1] = -vectors[:, 2]
    skew[:, 0, 2] = vectors[:, 1]
    skew[:, 1, 0] = vectors[:, 2]
    skew[:, 1, 2] = -vectors[:, 0]
    skew[:, 2, 0] = -vectors[:, 1]
    skew[:, 2, 1] = vectors[:, 0]
    return skew


def ground_penetration(
    body_q: torch.Tensor,  # [body_count, 7, 1]
    contact_mask: torch.Tensor,  # [body_count, max_contacts]
    contact_points: torch.Tensor,  # [body_count, max_contacts, 3, 1]
    ground_points: torch.Tensor,  # [body_count, max_contacts, 3, 1]
    contact_normals: torch.Tensor,  # [body_count, max_contacts, 3, 1]
) -> torch.Tensor:
    body_count = body_q.shape[0]
    max_contacts = contact_points.shape[1]

    points_world = transform_points_batch(
        contact_points, body_q.expand(body_count, max_contacts, 7, 1)
    )
    diff = ground_points - points_world

    penetration = (diff * contact_normals).sum(dim=2)

    # Zero out the penetration where the mask is false
    penetration[~contact_mask] = 0.0

    return penetration


def dC_penetration(
    body_q_a: torch.Tensor,
    body_qd_a: torch.Tensor,
    body_q_b: torch.Tensor,
    body_qd_b: torch.Tensor,
    r_a: torch.Tensor,
    r_b: torch.Tensor,
    n: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    J_a, J_b = J_penetration(body_q_a, body_q_b, r_a, r_b, n)  # ([N, 6], [N, 6])
    dC_a = (J_a * body_qd_a).sum(dim=1)  # [N]
    dC_b = (J_b * body_qd_b).sum(dim=1)  # [N]
    return dC_a, dC_b


def J_penetration(
    body_q_a: torch.Tensor,
    body_q_b: torch.Tensor,
    r_a: torch.Tensor,
    r_b: torch.Tensor,
    n: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r_a_world = rotate_vectors_batch(r_a, body_q_a)  # [N, 3]
    r_b_world = rotate_vectors_batch(r_b, body_q_b)  # [N, 3]

    r_a_x_n = torch.linalg.cross(r_a_world, n, dim=1)  # [N, 3]
    r_b_x_n = torch.linalg.cross(r_b_world, n, dim=1)  # [N, 3]

    J_a = torch.cat((-r_a_x_n, -n), dim=1)  # [N, 6]
    J_b = torch.cat((r_b_x_n, n), dim=1)  # [N, 6]

    return J_a, J_b


def compute_contact_jacobian(state):
    """
    Compute the batched contact Jacobian J_normal for all bodies and their contacts.

    Args:
        model: The simulation model with body transformations (body_q).
        state: The simulation state with contact data (contact_points, contact_normals, contact_mask).

    Returns:
        torch.Tensor: Jacobian of shape [body_count, max_contacts_per_body, 6].
    """
    body_count = state.body_q.shape[0]
    max_contacts = state.contact_mask.shape[1]
    device = state.body_q.device

    # Initialize the Jacobian with zeros
    J_normal = torch.zeros((body_count, max_contacts, 6), device=device)

    for b in range(body_count):
        # Get valid contacts for this body
        valid_contacts = state.contact_mask[b]  # [max_contacts]
        num_valid = valid_contacts.sum().item()

        if num_valid > 0:
            # Extract data for valid contacts
            local_points = state.contact_points[b, valid_contacts]  # [num_valid, 3, 1]
            normals = state.contact_normals[b, valid_contacts]  # [num_valid, 3, 1]

            # Transform local points to world coordinates
            body_transforms = state.body_q[b].expand(
                num_valid, 7, 1
            )  # [num_valid, 7, 1]
            world_points = transform_points_batch(
                local_points, body_transforms
            )  # [num_valid, 3, 1]

            # Compute r = world_points - body_position
            body_pos = state.body_q[b, :3].view(1, 3, 1)  # [1, 3, 1]
            r = world_points - body_pos  # [num_valid, 3, 1]

            # Compute r x n
            r_cross_n = torch.cross(r, normals, dim=1)  # [num_valid, 3, 1]

            # Assemble Jacobian: [-r_cross_n, -normals]
            J = torch.cat([r_cross_n, normals], dim=1)  # [num_valid, 6, 1]

            # Assign to the output tensor
            J_normal[b, valid_contacts] = J.squeeze(2)

    return J_normal


def contact_relative_velocity(
    body_q_a: torch.Tensor,
    body_qd_a: torch.Tensor,
    body_q_b: torch.Tensor,
    body_qd_b: torch.Tensor,
    r_a: torch.Tensor,
    r_b: torch.Tensor,
    n: torch.Tensor,
) -> torch.Tensor:
    # Linear velocities
    v_a = body_qd_a[3:]  # [N, 3]
    v_b = body_qd_b[3:]  # [N, 3]
    # Angular velocities
    w_a = body_qd_a[:3]  # [N, 3]
    w_b = body_qd_b[:3]  # [N, 3]
    # Rotated position vectors
    r_a_world = rotate_vectors_batch(r_a, body_q_a)  # [N, 3]
    r_b_world = rotate_vectors_batch(r_b, body_q_b)  # [N, 3]
    # Velocity due to rotation
    w_a_x_r_a = torch.linalg.cross(w_a, r_a_world, dim=1)  # [N, 3]
    w_b_x_r_b = torch.linalg.cross(w_b, r_b_world, dim=1)  # [N, 3]
    # Total relative velocity
    v_rel = v_b + w_b_x_r_b - v_a - w_a_x_r_a  # [N, 3]
    return v_rel


def compute_terrain_friction_jacobian(state):
    body_count = state.body_q.shape[0]
    max_contacts = state.contact_mask.shape[1]
    device = state.body_q.device

    # Initialize the Jacobian with zeros
    J_friction = torch.zeros((body_count, max_contacts, 2, 6), device=device)

    for b in range(body_count):
        # Get valid contacts for this body
        valid_contacts = state.contact_mask[b]  # [max_contacts]
        num_valid = valid_contacts.sum().item()

        if num_valid > 0:
            # Extract data for valid contacts
            local_points = state.contact_points[b, valid_contacts]  # [num_valid, 3]
            normals = state.contact_normals[b, valid_contacts]  # [num_valid, 3]

            # Transform local points to world coordinates
            body_transforms = state.body_q[b].expand(num_valid, -1)  # [num_valid, 7]
            world_points = transform_points_batch(
                local_points, body_transforms
            )  # [num_valid, 3]

            # Compute r = world_points - body_position
            body_pos = state.body_q[b, :3].unsqueeze(0)  # [1, 3]
            r = world_points - body_pos  # [num_valid, 3]


def J_friction(
    body_q_a: torch.Tensor,
    body_qd_a: torch.Tensor,
    body_q_b: torch.Tensor,
    body_qd_b: torch.Tensor,
    r_a: torch.Tensor,
    r_b: torch.Tensor,
    n: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    v_rel = contact_relative_velocity(
        body_q_a, body_qd_a, body_q_b, body_qd_b, r_a, r_b, n
    )  # [N, 3]
    v_rel = v_rel / (torch.linalg.norm(v_rel, dim=1, keepdim=True) + 1e-6)  # [N, 3]
    v_n = torch.bmm(v_rel.unsqueeze(1), n.unsqueeze(2)).squeeze()  # [N]

    v_t = v_rel - v_n * n  # [N, 3]

    t = v_t / (torch.linalg.norm(v_t, dim=1, keepdim=True) + 1e-6)  # [N, 3]
    b = torch.linalg.cross(n, t, dim=1)  # [N, 3]

    r_a_world = rotate_vectors_batch(r_a, body_q_a)  # [N, 3]
    r_b_world = rotate_vectors_batch(r_b, body_q_b)  # [N, 3]

    r_a_x_t = torch.linalg.cross(r_a_world, t, dim=1)  # [N, 3]
    r_b_x_t = torch.linalg.cross(r_b_world, t, dim=1)  # [N, 3]
    r_a_x_b = torch.linalg.cross(r_a_world, b, dim=1)  # [N, 3]
    r_b_x_b = torch.linalg.cross(r_b_world, b, dim=1)  # [N, 3]

    J_a_t = torch.cat((-r_a_x_t, -t), dim=1)  # [N, 6]
    J_b_t = torch.cat((r_b_x_t, t), dim=1)  # [N, 6]

    J_a_b = torch.cat((-r_a_x_b, -b), dim=1)  # [N, 6]
    J_b_b = torch.cat((r_b_x_b, b), dim=1)  # [N, 6]

    J_a = torch.cat((J_a_t.unsqueeze(1), J_a_b.unsqueeze(1)), dim=1)  # [N, 2, 6]
    J_b = torch.cat((J_b_t.unsqueeze(1), J_b_b.unsqueeze(1)), dim=1)  # [N, 2, 6]

    return J_a, J_b


def dC_friction(
    body_q_a: torch.Tensor,
    body_qd_a: torch.Tensor,
    body_q_b: torch.Tensor,
    body_qd_b: torch.Tensor,
    r_a: torch.Tensor,
    r_b: torch.Tensor,
    n: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    J_a, J_b = J_friction(
        body_q_a, body_qd_a, body_q_b, body_qd_b, r_a, r_b, n
    )  # ([N, 2, 6], [N, 2, 6])
    dC_a = (J_a * body_qd_a.unsqueeze(1)).sum(dim=2)  # [N, 2]
    dC_b = (J_b * body_qd_b.unsqueeze(1)).sum(dim=2)  # [N, 2]
    return dC_a, dC_b


def C_revolute(
    body_q: torch.Tensor,
    joint_parent: torch.Tensor,
    joint_child: torch.Tensor,
    joint_X_p: torch.Tensor,
    joint_X_c: torch.Tensor,
    joint_basis: torch.Tensor,
) -> torch.Tensor:
    # Axis of rotation is the z-axis of the joint frame
    device = body_q.device
    joint_count = joint_parent.shape[0]

    parent_q = body_q[joint_parent]  # [N, 4]
    child_q = body_q[joint_child]  # [N, 4]

    # Joint frame from the parent body
    X_wj_p = transform_multiply_batch(parent_q, joint_X_p)  # [N, 7]

    # Joint frame from the child body
    X_wj_c = transform_multiply_batch(child_q, joint_X_c)  # [N, 7]

    # Origin of the joint frame from the parent body
    o_parent = X_wj_p[:, :3]  # [N, 3]

    # Origin of the joint frame from the child body
    o_child = X_wj_c[:, :3]  # [N, 3]

    # Translational constraint: joint origins must coincide
    C_trans = o_parent - o_child  # [N, 3]

    # Rotational constraints: x and y of child perpendicular to z of parent    x_axis = torch.tensor([1.0, 0.0, 0.0], device=device).repeat(
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=device).repeat(
        joint_count, 1
    )  # [N, 3]
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device).repeat(
        joint_count, 1
    )  # [N, 3]
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device).repeat(
        joint_count, 1
    )  # [N, 3]

    x_axis_child = rotate_vectors_batch(x_axis, X_wj_c)  # [N, 3]
    y_axis_child = rotate_vectors_batch(y_axis, X_wj_c)  # [N, 3]
    z_axis_parent = rotate_vectors_batch(z_axis, X_wj_p)  # [N, 3]

    x_constraint = (x_axis_child * z_axis_parent).sum(dim=1)  # [N]
    y_constraint = (y_axis_child * z_axis_parent).sum(dim=1)  # [N]
    C_rot = torch.stack((x_constraint, y_constraint), dim=1)  # [N, 2]

    C = torch.cat([C_trans, C_rot], dim=1)  # [N, 5]
    return C


def dC_revolute(
    body_q: torch.Tensor,
    body_qd: torch.Tensor,
    joint_parent: torch.Tensor,
    joint_child: torch.Tensor,
    joint_X_p: torch.Tensor,
    joint_X_c: torch.Tensor,
    joint_basis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    J_p_trans, J_c_trans = J_translation(
        body_q, joint_parent, joint_child, joint_X_p, joint_X_c, joint_basis
    )  # ([N, 3, 6], [N, 3, 6])
    J_p_rot, J_c_rot = J_rotation(
        body_q, joint_parent, joint_child, joint_X_p, joint_X_c, joint_basis
    )  # ([N, 2, 6], [N, 2, 6])

    # Stack the translation and rotation Jacobians
    J_p = torch.cat([J_p_trans, J_p_rot], dim=1)  # [N, 5, 6]
    J_c = torch.cat([J_c_trans, J_c_rot], dim=1)  # [N, 5, 6]

    body_qd_p = body_qd[joint_parent]  # [N, 6]
    body_qd_c = body_qd[joint_child]  # [N, 6]
    dC_p = (J_p * body_qd_p.unsqueeze(1)).sum(dim=2)  # [N, 5]
    dC_c = (J_c * body_qd_c.unsqueeze(1)).sum(dim=2)  # [N, 5]
    return dC_p, dC_c


def J_translation(
    body_q: torch.Tensor,
    joint_parent: torch.Tensor,
    joint_child: torch.Tensor,
    joint_X_p: torch.Tensor,
    joint_X_c: torch.Tensor,
    joint_basis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = body_q.device
    joint_count = joint_parent.shape[0]
    E = torch.eye(3, device=device).repeat(joint_count, 1, 1)  # [N, 3, 3]

    parent_q = body_q[joint_parent]  # [N, 4]
    child_q = body_q[joint_child]  # [N, 4]

    # Joint frame from the parent body
    X_wj_p = transform_multiply_batch(parent_q, joint_X_p)  # [N, 7]
    X_wj_c = transform_multiply_batch(child_q, joint_X_c)  # [N, 7]

    r_p = X_wj_p[:, :3] - parent_q[:, :3]  # [N, 3]
    r_c = X_wj_c[:, :3] - child_q[:, :3]  # [N, 3]

    r_px = skew_symmetric_matrix_batch(r_p)  # [N, 3, 3]
    r_cx = skew_symmetric_matrix_batch(r_c)  # [N, 3, 3]

    J_p = torch.cat([r_px, -E], dim=1)  # [N, 3, 6]
    J_c = torch.cat([r_cx, E], dim=1)  # [N, 3, 6]

    return J_p, J_c


def J_rotation(
    body_q: torch.Tensor,
    joint_parent: torch.Tensor,
    joint_child: torch.Tensor,
    joint_X_p: torch.Tensor,
    joint_X_c: torch.Tensor,
    joint_basis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = body_q.device
    joint_count = joint_parent.shape[0]
    parent_q = body_q[joint_parent]  # [N, 4]
    child_q = body_q[joint_child]  # [N, 4]

    X_wj_p = transform_multiply_batch(parent_q, joint_X_p)  # [N, 7]
    X_wj_c = transform_multiply_batch(child_q, joint_X_c)  # [N, 7]

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=device).repeat(
        joint_count, 1
    )  # [N, 3]
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device).repeat(
        joint_count, 1
    )  # [N, 3]
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device).repeat(
        joint_count, 1
    )  # [N, 3]

    x_axis_child = rotate_vectors_batch(x_axis, X_wj_c)  # [N, 3]
    y_axis_child = rotate_vectors_batch(y_axis, X_wj_c)  # [N, 3]
    z_axis_parent = rotate_vectors_batch(z_axis, X_wj_p)  # [N, 3]

    x_c_x_z_p = torch.linalg.cross(x_axis_child, z_axis_parent, dim=1)  # [N, 3]
    y_c_x_z_p = torch.linalg.cross(y_axis_child, z_axis_parent, dim=1)  # [N, 3]

    J_p = torch.zeros(joint_count, 2, 6, device=device)  # [N, 2, 6]
    J_p[:, 0, :3] = -x_c_x_z_p
    J_p[:, 1, :3] = -y_c_x_z_p

    J_c = torch.zeros(joint_count, 2, 6, device=device)  # [N, 2, 6]
    J_c[:, 0, :3] = x_c_x_z_p
    J_c[:, 1, :3] = y_c_x_z_p

    return J_p, J_c
