import torch
from pbd_torch.model import Model
from pbd_torch.model import State
from pbd_torch.transform import *


def collide(model: Model, state: State):
    contact_points = []
    contact_normals = []
    contact_body_indices = []

    for b in range(model.body_count):
        transform = state.body_q[b].flatten()
        coll_points = model.body_collision_points[b].squeeze(0)
        x, q = transform[:3], transform[3:]

        transformed_points = rotate_vectors(coll_points, q) + x

        # Check if any points are below the ground
        below_ground = transformed_points[:, 2] < 0

        if below_ground.any():
            # Get the contact points
            body_points = coll_points[below_ground]
            body_normals = torch.tensor([0.0, 0.0,
                                         -1]).repeat(len(body_points), 1)
            body_indices = torch.full((len(body_points), ), b)

            contact_points.append(body_points)
            contact_normals.append(body_normals)
            contact_body_indices.append(body_indices)

    if len(contact_points) > 0:
        state.contact_point = torch.cat(contact_points, dim=0)
        state.contact_normal = torch.cat(contact_normals, dim=0)
        state.contact_body = torch.cat(contact_body_indices, dim=0)
        state.contact_count = len(state.contact_point)