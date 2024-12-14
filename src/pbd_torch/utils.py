import os
import json

from typing import Union, List, NewType
from dataclasses import dataclass

import torch
import matplotlib.pyplot as plt

from pbd_torch.transform import rotate_vectors
from pbd_torch.constants import *
from pbd_torch.model import Model, State


# Define new types for the simulation
Vector3 = NewType('Vector3', torch.Tensor)  # (3, ) tensor
Transform = NewType('Transform', torch.Tensor)  # (7, ) tensor
Matrix3x3 = NewType('Matrix3x3', torch.Tensor)  # (3, 3) tensor
Quaternion = NewType('Quaternion', torch.Tensor)  # (4, ) tensor

@dataclass
class Collision:
    point: torch.Tensor
    normal: torch.Tensor

@dataclass
class GlobalVector:
    """
    Represents a vector in global (world) coordinates.
    This class is primarily used for plotting purposes and defines a vector by its:

    - Origin: The starting point of the vector in global coordinates.
    - Destination: The ending point of the vector in global coordinates.

    Attributes:
        origin (torch.Tensor): A 3D point representing the starting position in global coordinates.
        destination (torch.Tensor): A 3D point representing the ending position in global coordinates.
    """
    origin: torch.Tensor
    destination: torch.Tensor


@dataclass
class LocalVector:
    """
    Represents a vector in local coordinates.
    This class is used for plotting and includes information about the vector's:

    - r: The vector itself in local coordinates.
    - x: The translation of the local coordinate frame relative to the global (world) coordinates.
    - q: The rotation of the local coordinate frame relative to the global (world) coordinates.
    - color: The color of the vector when plotted.

    Attributes:
        r (torch.Tensor): A vector in local coordinate space.
        x (torch.Tensor): The translation of the local frame with respect to the global frame.
        q (torch.Tensor): The quaternion representing the rotation of the local frame with respect to the global frame.
        color (Union[str, tuple]): The color of the vector when plotted.
    """
    r: torch.Tensor
    x: torch.Tensor
    q: torch.Tensor
    color: Union[str, tuple] = ORANGE


def plot_axis(ax: plt.Axes, x: torch.Tensor, q: torch.Tensor, scale: float = 1.0):
    x_axis = torch.tensor([1.0, 0.0, 0.0]) * scale
    y_axis = torch.tensor([0.0, 1.0, 0.0]) * scale
    z_axis = torch.tensor([0.0, 0.0, 1.0]) * scale

    x_rot = rotate_vectors(x_axis, q)
    y_rot = rotate_vectors(y_axis, q)
    z_rot = rotate_vectors(z_axis, q)

    ax.quiver(x[0], x[1], x[2], x_rot[0], x_rot[1], x_rot[2], color='r')
    ax.quiver(x[0], x[1], x[2], y_rot[0], y_rot[1], y_rot[2], color='g')
    ax.quiver(x[0], x[1], x[2], z_rot[0], z_rot[1], z_rot[2], color='b')


def plot_collisions(ax: plt.Axes,
                    x: torch.tensor,
                    q: torch.tensor,
                    collisions: List[Collision],
                    color: Union[str, tuple] = 'r'):
    for collision in collisions:
        r = collision.point

        # Transform collision point to world coordinates
        c = rotate_vectors(r, q) + x

        # Plot collision point
        ax.scatter(c[0], c[1], c[2], c=color, marker='x')


def plot_box_geometry(ax: plt.Axes,
                      x: torch.Tensor,
                      q: torch.Tensor,
                      dims: torch.Tensor,
                      color: Union[str, tuple] = 'b'):
    hx, hy, hz = dims[0] / 2, dims[1] / 2, dims[2] / 2
    corners = torch.tensor([[-hx, -hy, -hz],
                            [hx, -hy, -hz],
                            [hx, hy, -hz],
                            [-hx, hy, -hz],
                            [-hx, -hy, hz],
                            [hx, -hy, hz],
                            [hx, hy, hz],
                            [-hx, hy, hz]])

    rotated_corners = rotate_vectors(corners, q) + x
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for edge in edges:
        start, end = rotated_corners[edge[0]], rotated_corners[edge[1]]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color)

def plot_vectors(ax: plt.Axes, vectors: Union[List[LocalVector], List[GlobalVector]], color: Union[str, tuple] = 'r'):
    for vector in vectors:
        if isinstance(vector, LocalVector):
            v = rotate_vectors(vector.r, vector.q)
            ax.quiver(vector.x[0], vector.x[1], vector.x[2], v[0], v[1], v[2], color=vector.color)
        elif isinstance(vector, GlobalVector):
            print(f"Origin: {vector.origin}, Destination: {vector.destination}")
            dv = vector.destination - vector.origin
            ax.quiver(vector.origin[0], vector.origin[1], vector.origin[2],
                        dv[0], dv[1], dv[2], color=color)
        else:
            raise ValueError("Invalid vector type. Must be either LocalVector or GlobalVector.")

def save_simulation(model: Model, states: List[State], output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    simulation = {
        'model': model.serialize(),
        'states': [state.serialize(model) for state in states]
    }

    json_simulation = json.dumps(simulation, indent=2)
    with open(output_file, 'w') as f:
        f.write(json_simulation)