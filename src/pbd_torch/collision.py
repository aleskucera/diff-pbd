import torch
from pbd_torch.model import Model
from pbd_torch.model import State
from pbd_torch.transform import *


def collide(model: Model, state: State, collision_margin: float = 0.0):
    if model.terrain is not None:
        collide_terrain(model, state, collision_margin)
    else:
        collide_ground(model, state, collision_margin)


def collide_ground(model: Model, state: State, collision_margin: float = 0.0):
    contact_points = []
    contact_normals = []
    contact_body_indices = []
    contact_points_indices = []

    for b in range(model.body_count):
        transform = state.body_q[b].flatten()
        coll_points = model.body_collision_points[b].squeeze(0)
        x, q = transform[:3], transform[3:]

        transformed_points = rotate_vectors(coll_points, q) + x

        # Check if any points are below the ground
        below_ground = transformed_points[:, 2] < collision_margin

        if below_ground.any():
            # Get the contact points
            body_points = coll_points[below_ground]
            body_normals = torch.tensor([0.0, 0.0,
                                         -1.0], device=model.device).repeat(len(body_points), 1)
            body_indices = torch.full((len(body_points),), b)

            contact_points.append(body_points)
            contact_normals.append(body_normals)
            contact_body_indices.append(body_indices)
            contact_points_indices.append(below_ground.nonzero())

    if len(contact_points) > 0:
        model.contact_point = torch.cat(contact_points, dim=0)
        model.contact_normal = torch.cat(contact_normals, dim=0)
        model.contact_body = torch.cat(contact_body_indices, dim=0)
        model.contact_point_idx = torch.cat(contact_points_indices, dim=0)
        model.contact_count = len(model.contact_point)


def collide_terrain(model: Model, state: State, collision_margin: float = 0.0):
    contact_points = []
    contact_normals = []
    contact_body_indices = []
    contact_points_indices = []
    
    for b in range(model.body_count):
        transform = state.body_q[b].flatten()
        coll_points = model.body_collision_points[b].squeeze(0)
        x, q = transform[:3], transform[3:]
        transformed_points = rotate_vectors(coll_points, q) + x

        ground_height = model.terrain.get_height_at_point(transformed_points[:, 0], transformed_points[:, 1])
        ground_normal = model.terrain.get_normal_at_point(transformed_points[:, 0], transformed_points[:, 1])

        # Check if any points are below the ground
        below_ground = transformed_points[:, 2] - ground_height < collision_margin
        
        if below_ground.any():
            # Get the contact points
            body_points = coll_points[below_ground]
            body_normals = ground_normal[below_ground]
            body_indices = torch.full((len(body_points), ), b)
            
            contact_points.append(body_points)
            contact_normals.append(body_normals)
            contact_body_indices.append(body_indices)
            contact_points_indices.append(below_ground.nonzero())

    if len(contact_points) > 0:
        model.contact_point = torch.cat(contact_points, dim=0)
        model.contact_normal = torch.cat(contact_normals, dim=0)
        model.contact_body = torch.cat(contact_body_indices, dim=0)
        model.contact_point_idx = torch.cat(contact_points_indices, dim=0)
        model.contact_count = len(model.contact_point)
