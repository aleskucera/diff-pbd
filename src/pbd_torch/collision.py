import torch
from pbd_torch.model import Model
from pbd_torch.model import State
from pbd_torch.transform import transform_points_batch


def collide(model: Model, state: State, collision_margin: float = 0.0):
    if model.terrain is not None:
        collide_terrain(model, state, collision_margin)
    else:
        collide_ground(model, state, collision_margin)


def smooth_step_function(
    x: torch.Tensor, alpha: float = 200.0, margin: float = 0.0
) -> torch.Tensor:
    out = 1 - torch.sigmoid(alpha * (x - margin))
    return out


def collide_ground(model: Model, state: State, collision_margin: float = 0.0):
    device = model.device
    B = model.body_count
    C = model.max_contacts_per_body
    P = model.coll_points.shape[0]  # Total collision points across all bodies

    points = model.coll_points  # [P, 3, 1]
    body_indices = model.coll_points_body_idx  # [P]

    # Transform collision points to world coordinates
    body_transforms = state.body_q[body_indices]  # [P, 7, 1]
    points_world = transform_points_batch(points, body_transforms)  # [P, 3, 1]

    z = points_world[:, 2, 0]  # [P], extract z-coordinates

    # Create contact mask for points below the ground plane (z < collision_margin)
    contact_mask = z < collision_margin  # [P]

    # Filter the contact data
    contact_points = points[contact_mask]  # [n_all_contact_points, 3, 1]
    contact_body_indices = body_indices[contact_mask]  # [n_all_contact_points]
    points_world_contact = points_world[contact_mask]  # [n_all_contact_points, 3, 1]
    height_above_ground = z[contact_mask]  # [n_all_contact_points]

    # Ground points: same x and y as contact points, z = 0
    ground_points = points_world_contact.clone()  # [n_all_contact_points, 3, 1]
    ground_points[:, 2, 0] = 0.0  # Set z to 0 for the ground plane

    # Contact normals: upward direction [0, 0, 1]
    contact_normals = torch.zeros_like(
        points_world_contact
    )  # [n_all_contact_points, 3, 1]
    contact_normals[:, 2, 0] = 1.0  # Set normals to [0, 0, 1]

    # Count points per body and create position indices
    counts = torch.bincount(contact_body_indices, minlength=B)  # [B]
    cum_counts = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)[:-1]]
    )  # [B]
    point_pos = (
        torch.arange(len(contact_body_indices), dtype=torch.long, device=device)
        - cum_counts[contact_body_indices]
    )  # [n_all_contact_points]

    # Filter to include only up to C contacts per body
    valid = point_pos < C

    # Initialize batched tensors in state
    state.contact_mask_per_body = torch.zeros((B, C), dtype=torch.bool, device=device)
    state.contact_weight_per_body = torch.zeros((B, C), dtype=torch.float32, device=device)
    state.contact_point_indices_per_body = torch.zeros(
        (B, C), dtype=torch.long, device=device
    )
    state.contact_points_per_body = torch.zeros(
        (B, C, 3, 1), dtype=torch.float32, device=device
    )
    state.contact_normals_per_body = torch.zeros(
        (B, C, 3, 1), dtype=torch.float32, device=device
    )
    state.contact_points_ground_per_body = torch.zeros(
        (B, C, 3, 1), dtype=torch.float32, device=device
    )

    # Scatter contact data for valid contacts
    valid_body_indices = contact_body_indices[valid]
    valid_point_pos = point_pos[valid]

    weight = smooth_step_function(height_above_ground, alpha=200.0)

    state.contact_mask_per_body[valid_body_indices, valid_point_pos] = True
    state.contact_weight_per_body[valid_body_indices, valid_point_pos] = weight[valid]
    state.contact_point_indices_per_body[valid_body_indices, valid_point_pos] = (
        torch.arange(valid.sum(), device=device)
    )
    state.contact_points_per_body[valid_body_indices, valid_point_pos] = contact_points[
        valid
    ]
    state.contact_normals_per_body[valid_body_indices, valid_point_pos] = (
        contact_normals[valid]
    )
    state.contact_points_ground_per_body[valid_body_indices, valid_point_pos] = (
        ground_points[valid]
    )

    # Assign flat contact attributes
    n_contacts = valid.sum().item()
    state.contact_count = n_contacts
    state.contact_body_indices_flat = contact_body_indices[valid]  # [P]
    state.contact_points_flat = contact_points[valid]  # [P, 3, 1]
    state.contact_normals_flat = contact_normals[valid]  # [P, 3, 1]
    state.contact_points_ground_flat = ground_points[valid]  # [P, 3, 1]


def collide_terrain(model: Model, state: State, collision_margin: float = 0.0):
    device = model.device
    B = model.body_count  # Number of bodies
    C = model.max_contacts_per_body  # Maximum number of contact points per body
    P = model.coll_points.shape[0]  # All body points in the model

    points = model.coll_points  # [P, 3, 1]
    body_indices = model.coll_points_body_idx  # [P]

    # Transform collision points to world coordinates
    body_transforms = state.body_q[body_indices]  # [P, 7, 1]
    points_world = transform_points_batch(points, body_transforms)  # [P, 3, 1]

    x = points_world[:, 0]  # [P, 1]
    y = points_world[:, 1]  # [P, 1]
    z = points_world[:, 2]  # [P, 1]

    # Get terrain height and normal
    ground_height = model.terrain.get_height_at_point(x, y)  # [P, 1]
    ground_points = torch.stack([x, y, ground_height], dim=1)  # [P, 3, 1]
    ground_normal = model.terrain.get_normal_at_point(x, y).transpose(1, 2)  # [P, 3, 1]

    # Create contact mask
    contact_mask = ((z - ground_height) < collision_margin).flatten()  # [P]
    contact_weight = smooth_step_function((z - ground_height)[contact_mask].flatten(), alpha=200)  # [n_all_contact_points]

    # Filter the contact data
    contact_points = points[contact_mask]  # [n_all_contact_points, 3, 1]
    contact_body_indices = body_indices[contact_mask]  # [n_all_contact_points]
    contact_normals = ground_normal[contact_mask]  # [n_all_contact_points, 3, 1]
    ground_points = ground_points[contact_mask]  # [n_all_contact_points, 3, 1]

    # Count points per body and create position indices
    counts = torch.bincount(contact_body_indices, minlength=B)  # [B]
    cum_counts = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)[:-1]]
    )  # [B]
    point_pos = (
        torch.arange(len(contact_body_indices), dtype=torch.long, device=device)
        - cum_counts[contact_body_indices]
    )  # [n_all_contact_points]

    # Filter to include only up to C contacts per body
    valid = point_pos < C  # [n_all_contact_points]

    # Initialize batched tensors in state
    state.contact_mask_per_body = torch.zeros((B, C), dtype=torch.bool, device=device)
    state.contact_weight_per_body = torch.zeros((B, C), dtype=torch.float32, device=device)
    state.contact_point_indices_per_body = torch.zeros(
        (B, C), dtype=torch.long, device=device
    )
    state.contact_points_per_body = torch.zeros(
        (B, C, 3, 1), dtype=torch.float32, device=device
    )
    state.contact_normals_per_body = torch.zeros(
        (B, C, 3, 1), dtype=torch.float32, device=device
    )
    state.contact_points_ground_per_body = torch.zeros(
        (B, C, 3, 1), dtype=torch.float32, device=device
    )

    # Scatter contact data
    valid_body_indices = contact_body_indices[valid]
    valid_point_pos = point_pos[valid]

    state.contact_mask_per_body[valid_body_indices, valid_point_pos] = True
    state.contact_weight_per_body[valid_body_indices, valid_point_pos] = contact_weight[valid]
    state.contact_point_indices_per_body[valid_body_indices, valid_point_pos] = (
        contact_body_indices[valid]
    )
    state.contact_points_per_body[valid_body_indices, valid_point_pos] = contact_points[
        valid
    ]
    state.contact_normals_per_body[valid_body_indices, valid_point_pos] = (
        contact_normals[valid]
    )
    state.contact_points_ground_per_body[valid_body_indices, valid_point_pos] = (
        ground_points[valid]
    )

    # Assign flat contact attributes
    n_contacts = valid.sum().item()
    state.contact_count = n_contacts
    state.contact_body_indices_flat = contact_body_indices[valid]  # [P]
    state.contact_points_flat = contact_points[valid]  # [P, 3, 1]
    state.contact_normals_flat = contact_normals[valid]  # [P, 3, 1]
    state.contact_points_ground_flat = ground_points[valid]  # [P, 3, 1]

def collide_terrain_friction(model: Model, state: State, collision_margin: float = 0.0):
    device = model.device
    B = model.body_count  # Number of bodies
    C = model.max_contacts_per_body  # Maximum number of contact points per body
    P = model.coll_points.shape[0]  # All body points in the model

    points = model.coll_points  # [P, 3, 1]
    body_indices = model.coll_points_body_idx  # [P]

    # Transform collision points to world coordinates
    body_transforms = state.body_q[body_indices]  # [P, 7, 1]
    points_world = transform_points_batch(points, body_transforms)  # [P, 3, 1]

    x = points_world[:, 0]  # [P, 1]
    y = points_world[:, 1]  # [P, 1]
    z = points_world[:, 2]  # [P, 1]

    # Get terrain height and normal
    ground_height = model.terrain.get_height_at_point(x, y)  # [P, 1]
    ground_points = torch.stack([x, y, ground_height], dim=1)  # [P, 3, 1]
    ground_normal = model.terrain.get_normal_at_point(x, y).transpose(1, 2)  # [P, 3, 1]

    # Create contact mask
    contact_mask = ((z - ground_height) < collision_margin).flatten()  # [P]
    contact_weight = smooth_step_function((z - ground_height)[contact_mask].flatten(), alpha=200)  # [n_all_contact_points]

    # Filter the contact data
    contact_points = points[contact_mask]  # [n_all_contact_points, 3, 1]
    contact_body_indices = body_indices[contact_mask]  # [n_all_contact_points]
    contact_normals = ground_normal[contact_mask]  # [n_all_contact_points, 3, 1]
    ground_points = ground_points[contact_mask]  # [n_all_contact_points, 3, 1]

    # Count points per body and create position indices
    counts = torch.bincount(contact_body_indices, minlength=B)  # [B]
    cum_counts = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)[:-1]]
    )  # [B]
    point_pos = (
            torch.arange(len(contact_body_indices), dtype=torch.long, device=device)
            - cum_counts[contact_body_indices]
    )  # [n_all_contact_points]

    # Filter to include only up to C contacts per body
    valid = point_pos < C  # [n_all_contact_points]

    # Initialize batched tensors in state
    state.contact_mask_per_body_friction = torch.zeros((B, C), dtype=torch.bool, device=device)
    state.contact_weight_per_body_friction = torch.zeros((B, C), dtype=torch.float32, device=device)
    state.contact_point_indices_per_body_friction = torch.zeros(
        (B, C), dtype=torch.long, device=device
    )
    state.contact_points_per_body_friction = torch.zeros(
        (B, C, 3, 1), dtype=torch.float32, device=device
    )
    state.contact_normals_per_body_friction = torch.zeros(
        (B, C, 3, 1), dtype=torch.float32, device=device
    )
    state.contact_points_ground_per_body_friction = torch.zeros(
        (B, C, 3, 1), dtype=torch.float32, device=device
    )

    # Scatter contact data
    valid_body_indices = contact_body_indices[valid]
    valid_point_pos = point_pos[valid]

    state.contact_mask_per_body_friction[valid_body_indices, valid_point_pos] = True
    state.contact_weight_per_body_friction[valid_body_indices, valid_point_pos] = contact_weight[valid]
    state.contact_point_indices_per_body_friction[valid_body_indices, valid_point_pos] = (
        contact_body_indices[valid]
    )
    state.contact_points_per_body_friction[valid_body_indices, valid_point_pos] = contact_points[
        valid
    ]
    state.contact_normals_per_body_friction[valid_body_indices, valid_point_pos] = (
        contact_normals[valid]
    )
    state.contact_points_ground_per_body_friction[valid_body_indices, valid_point_pos] = (
        ground_points[valid]
    )

if __name__ == "__main__":
    # Parameters
    import matplotlib.pyplot as plt

    x = torch.linspace(-0.1, 0.1, 500)  # Input range
    alphas = [50.0, 100.0, 150.0, 200.0]  # Different alpha values
    margin = 0.0  # Margin

    # Plot
    plt.figure(figsize=(8, 6))
    for alpha in alphas:
        y = smooth_step_function(x, alpha, margin)
        plt.plot(x.numpy(), y.numpy(), label=f"alpha = {alpha}")

    # Customization
    plt.title("Smooth Step Function with Different Alphas")
    plt.xlabel("x")
    plt.ylabel("Output")
    plt.legend()
    plt.grid(True)
    plt.show()
