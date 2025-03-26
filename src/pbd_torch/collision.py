import torch
from pbd_torch.model import Model
from pbd_torch.model import State
from pbd_torch.transform import transform_points_batch


def collide(model: Model, state: State, collision_margin: float = 0.0):
    if model.terrain is not None:
        collide_terrain(model, state, collision_margin)
    else:
        collide_ground(model, state, collision_margin)


def collide_ground(model: Model, state: State, collision_margin: float = 0.0):
    points_world = transform_points_batch(
        model.coll_points, state.body_q[model.coll_points_body_idx]
    )
    contact_mask = points_world[:, 2] < collision_margin

    contact_count = int(contact_mask.sum().item())

    contact_points = model.coll_points[contact_mask]
    contact_normals = torch.tensor([0.0, 0.0, -1.0], device=model.device).repeat(
        contact_count, 1
    )
    contact_body_indices = model.coll_points_body_idx[contact_mask]
    contact_points_ground = torch.stack(
        [
            points_world[contact_mask, 0].view(-1),
            points_world[contact_mask, 1].view(-1),
            torch.zeros(contact_count, device=model.device),
        ],
        dim=1,
    )
    contact_points_indices = contact_mask.nonzero()

    model.contact_count = contact_count
    model.contact_point = contact_points
    model.contact_normal = contact_normals
    model.contact_body = contact_body_indices
    model.contact_point_idx = contact_points_indices
    model.contact_point_ground = contact_points_ground


def collide_terrain(model: Model, state: State, collision_margin: float = 0.0):
    points_world = transform_points_batch(
        model.coll_points, state.body_q[model.coll_points_body_idx]
    )

    # Get terrain height and normal for all collision points
    ground_height = model.terrain.get_height_at_point(
        points_world[:, 0], points_world[:, 1]
    )
    ground_normal = model.terrain.get_normal_at_point(
        points_world[:, 0], points_world[:, 1]
    )

    # Create contact mask for points below terrain
    contact_mask = (points_world[:, 2] - ground_height) < collision_margin

    contact_count = int(contact_mask.sum().item())

    contact_points = model.coll_points[contact_mask]
    contact_normals = -ground_normal[contact_mask]
    contact_body_indices = model.coll_points_body_idx[contact_mask]
    contact_points_ground = torch.stack(
        [
            points_world[contact_mask, 0].view(-1),
            points_world[contact_mask, 1].view(-1),
            ground_height[contact_mask].view(-1),
        ],
        dim=1,
    )
    contact_points_indices = contact_mask.nonzero()

    model.contact_count = contact_count
    model.contact_point = contact_points
    model.contact_normal = contact_normals
    model.contact_body = contact_body_indices
    model.contact_point_idx = contact_points_indices
    model.contact_point_ground = contact_points_ground


def collide_batch(model: Model, state: State, collision_margin: float = 0.0):
    if model.terrain is not None:
        collide_terrain_batch(model, state, collision_margin)
    else:
        collide_ground_batch(model, state, collision_margin)


def collide_ground_batch(model: Model, state: State, collision_margin: float = 0.0):
    device = model.device

    # Transform collision points to world coordinates
    body_transforms = state.body_q[model.coll_points_body_idx]  # [n_points, 7, 4]
    points_world = transform_points_batch(
        model.coll_points, body_transforms
    )  # [n_points, 3, 1]
    contact_mask = (points_world[:, 2] < collision_margin).view(-1)  # [n_coll_points]

    # Filter the contact data
    contact_points = model.coll_points[contact_mask]  # [n_points, 3, 1]
    contact_body_indices = model.coll_points_body_idx[contact_mask]  # [n_points]

    ground_height = torch.zeros_like(contact_points[:, 0], device=device)  # [n_points]

    # Compute the contact normals
    contact_normals = (
        torch.tensor([0.0, 0.0, 1.0], device=device)
        .repeat(contact_points.shape[0], 1)
        .unsqueeze(-1)
    )  # [n_points, 3, 1]

    # Compute number of contacts per body
    max_contacts = model.max_contacts_per_body

    mask_shape = (state.body_count, max_contacts)
    vectors_shape = (state.body_count, max_contacts, 3, 1)

    # Initialize batched tensors in state
    state.contact_mask = torch.zeros(mask_shape, dtype=torch.bool, device=device)
    state.contact_points = torch.zeros(
        vectors_shape, dtype=torch.float32, device=device
    )
    state.contact_normals = torch.zeros(
        vectors_shape, dtype=torch.float32, device=device
    )
    state.contact_point_indices = torch.zeros(
        mask_shape, dtype=torch.int32, device=device
    )
    state.contact_points_ground = torch.zeros(
        vectors_shape, dtype=torch.float32, device=device
    )

    # Fill batched tensors
    for b in range(state.body_count):
        body_contacts = contact_body_indices == b
        num_contacts = min(
            body_contacts.sum().item(), max_contacts
        )  # Truncate if exceeds max
        if num_contacts > 0:
            body_contact_points = contact_points[body_contacts]  # [num_contacts, 3, 1]
            body_contact_normals = contact_normals[
                body_contacts
            ]  # [num_contacts, 3, 1]
            contact_point_indices = contact_mask.nonzero(as_tuple=True)[0][
                body_contacts
            ]  # [num_contacts]
            ground_points = torch.stack(
                [
                    points_world[contact_mask][body_contacts, 0],
                    points_world[contact_mask][body_contacts, 1],
                    ground_height[body_contacts],
                ],
                dim=1,
            )  # [num_contacts, 3, 1]

            state.contact_mask[b, :num_contacts] = True
            state.contact_points[b, :num_contacts] = body_contact_points[:num_contacts]
            state.contact_normals[b, :num_contacts] = body_contact_normals[
                :num_contacts
            ]
            state.contact_points_ground[b, :num_contacts] = ground_points[:num_contacts]
            state.contact_point_indices[b, :num_contacts] = contact_point_indices[
                :num_contacts
            ]


def collide_terrain_batch(model: Model, state: State, collision_margin: float = 0.0):
    device = model.device

    # Transform collision points to world coordinates
    body_transforms = state.body_q[model.coll_points_body_idx]
    points_world = transform_points_batch(model.coll_points, body_transforms)

    # Get terrain height and normal
    ground_height = model.terrain.get_height_at_point(
        points_world[:, 0], points_world[:, 1]
    )
    ground_normal = model.terrain.get_normal_at_point(
        points_world[:, 0], points_world[:, 1]
    ).transpose(1, 2)

    # Create contact mask
    contact_mask = ((points_world[:, 2] - ground_height) < collision_margin).view(-1)

    # Filter the contact data
    contact_points = model.coll_points[contact_mask]
    contact_body_indices = model.coll_points_body_idx[contact_mask]
    contact_normals = ground_normal[contact_mask]  # [n_points, 3, 1]

    # Compute number of contacts per body
    max_contacts = model.max_contacts_per_body

    mask_shape = (state.body_count, max_contacts)
    vectors_shape = (state.body_count, max_contacts, 3, 1)

    # Initialize batched tensors in state
    state.contact_mask = torch.zeros(mask_shape, dtype=torch.bool, device=device)
    state.contact_points = torch.zeros(
        vectors_shape, dtype=torch.float32, device=device
    )
    state.contact_normals = torch.zeros(
        vectors_shape, dtype=torch.float32, device=device
    )
    state.contact_point_indices = torch.zeros(
        mask_shape, dtype=torch.int32, device=device
    )
    state.contact_points_ground = torch.zeros(
        vectors_shape, dtype=torch.float32, device=device
    )

    # Fill batched tensors
    for b in range(state.body_count):
        body_contacts = contact_body_indices == b
        num_contacts = min(
            body_contacts.sum().item(), max_contacts
        )  # Truncate if exceeds max
        if num_contacts > 0:
            body_contact_points = contact_points[body_contacts]
            body_contact_normals = contact_normals[body_contacts]
            contact_point_indices = contact_mask.nonzero(as_tuple=True)[0][
                body_contacts
            ]
            ground_points = torch.stack(
                [
                    points_world[contact_mask][body_contacts, 0],
                    points_world[contact_mask][body_contacts, 1],
                    ground_height[contact_mask][body_contacts],
                ],
                dim=1,
            )  # [num_contacts, 3, 1]

            state.contact_mask[b, :num_contacts] = True
            state.contact_points[b, :num_contacts] = body_contact_points[:num_contacts]
            state.contact_normals[b, :num_contacts] = body_contact_normals[
                :num_contacts
            ]
            state.contact_points_ground[b, :num_contacts] = ground_points[:num_contacts]
            state.contact_point_indices[b, :num_contacts] = contact_point_indices[
                :num_contacts
            ]
