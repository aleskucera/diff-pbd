import torch
from pbd_torch.model import Model
from pbd_torch.model import State
from pbd_torch.transform import transform_points_batch
from jaxtyping import Float, Bool

def collide(model: Model, state: State, collision_margin: float = 0.0):
    if model.terrain is not None:
        collide_terrain(model, state, collision_margin)
    else:
        collide_ground(model, state, collision_margin)


def collide_ground(model: Model, state: State, collision_margin: float = 0.0):
    device = model.device
    B = model.body_count
    C = model.max_contacts_per_body

    # Transform collision points to world coordinates
    body_transforms = state.body_q[model.coll_points_body_idx]  # [n_points, 7, 4]
    points_world = transform_points_batch(model.coll_points, body_transforms)  # [n_points, 3, 1]
    contact_mask = (points_world[:, 2] < collision_margin).view(-1)  # [n_contact_points]

    # Filter the contact data
    contact_points = model.coll_points[contact_mask]  # [n_contact_points, 3, 1]
    contact_body_indices = model.coll_points_body_idx[contact_mask]  # [n_contact_points]

    # Sort the contact points by body index
    sort_indices = torch.argsort(contact_body_indices)
    contact_points_sorted = contact_points[sort_indices]
    contact_body_indices_sorted = contact_body_indices[sort_indices]

    ground_height = torch.zeros_like(contact_points[:, 0], device=device)  # [n_contact_points]

    # Compute the contact normals
    contact_normals = (
        torch.tensor([0.0, 0.0, 1.0], device=device)
        .repeat(contact_points.shape[0], 1)
        .unsqueeze(-1)
    )  # [n_contact_points, 3, 1]

    mask_shape = (state.body_count, C)
    vectors_shape = (state.body_count, C, 3, 1)

    # Initialize batched tensors in state
    contact_mask = torch.zeros(mask_shape, dtype=torch.bool, device=device)
    contact_points = torch.zeros(vectors_shape, dtype=torch.float32, device=device)
    contact_normals = torch.zeros(vectors_shape, dtype=torch.float32, device=device)
    contact_point_indices = torch.zeros(mask_shape, dtype=torch.int32, device=device)
    contact_points_ground = torch.zeros(vectors_shape, dtype=torch.float32, device=device)

    # Fill batched tensors
    for b in range(state.body_count):
        body_contacts = contact_body_indices == b
        num_contacts = min(
            body_contacts.sum().item(), C
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


def collide_terrain(model: Model, state: State, collision_margin: float = 0.0):
    device = model.device

    B = model.body_count
    C = model.max_contacts_per_body
    P = model.coll_points.shape[0]  # All body points in the model

    points = model.coll_points # [P, 3, 1]
    body_indices = model.coll_points_body_idx # [P]

    # Transform collision points to world coordinates
    body_transforms = state.body_q[body_indices] # [P, 7, 1]
    points_world = transform_points_batch(points, body_transforms) # [P, 3, 1]

    x = points_world[:, 0] # [P, 1]
    y = points_world[:, 1] # [P, 1]
    z = points_world[:, 2] # [P, 1]

    # Get terrain height and normal
    ground_height = model.terrain.get_height_at_point(x, y) # [P, 1]
    ground_points = torch.stack([x, y, ground_height], dim=1) # [P, 3, 1]
    ground_normal = model.terrain.get_normal_at_point(x, y).transpose(1, 2) # [P, 3, 1]

    # Create contact mask
    contact_mask = ((z - ground_height) < collision_margin).flatten() # [P]

    # Filter the contact data
    contact_points = points[contact_mask] # [n_all_contact_points, 3, 1]
    contact_body_indices = body_indices[contact_mask] # [n_all_contact_points]
    contact_normals = ground_normal[contact_mask]  # [n_all_contact_points, 3, 1]
    ground_points = ground_points[contact_mask] # [n_all_contact_points, 3, 1]

    # Count points per body and create positions indices
    counts = torch.bincount(contact_body_indices, minlength=B) # [B]
    cum_counts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)[:-1]]) # [B+1]
    point_pos = torch.arange(len(contact_body_indices), dtype=torch.long, device=device) - cum_counts[contact_body_indices] # [n_all_contact_points]

    # Filter and scatter
    valid = point_pos < C

    # Initialize batched tensors in state
    state.contact_mask = torch.zeros((B, C), dtype=torch.bool, device=device)
    state.contact_point_indices = torch.zeros((B, C), dtype=torch.long, device=device)
    state.contact_points = torch.zeros((B, C, 3, 1), dtype=torch.float32, device=device)
    state.contact_normals = torch.zeros((B, C, 3, 1), dtype=torch.float32, device=device)
    state.contact_points_ground = torch.zeros((B, C, 3, 1), dtype=torch.float32, device=device)

    # Scatter contact data
    state.contact_mask[contact_body_indices[valid], point_pos[valid]] = True
    state.contact_point_indices[contact_body_indices[valid], point_pos[valid]] = contact_body_indices[valid]
    state.contact_points[contact_body_indices[valid], point_pos[valid]] = contact_points[valid]
    state.contact_normals[contact_body_indices[valid], point_pos[valid]] = contact_normals[valid]
    state.contact_points_ground[contact_body_indices[valid], point_pos[valid]] = ground_points[valid]
