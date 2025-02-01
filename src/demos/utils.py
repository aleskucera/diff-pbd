import json
import os
from typing import List
from typing import Union

import torch
from matplotlib.axes import Axes
from pbd_torch.model import Model
from pbd_torch.model import State
from pbd_torch.transform import rotate_vectors


def save_simulation(model: Model, states: List[State], output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    simulation = {
        'model': model.serialize(),
        'states': [state.serialize(model) for state in states]
    }

    json_simulation = json.dumps(simulation, indent=2)
    with open(output_file, 'w') as f:
        f.write(json_simulation)


def plot_points(ax: Axes, points: torch.Tensor, title: str):
    # Plot the collision points
    ax.scatter(points[:, 0],
               points[:, 1],
               points[:, 2],
               c=points[:, 2],
               marker='o',
               s=10,
               cmap='viridis',
               alpha=0.7)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)


def plot_frame(ax: Axes,
               frame: torch.Tensor,
               origin_color: Union[str, tuple],
               scale: float = 1.0,
               label: str = None):
    origin = frame[:3]

    # Define axes
    x_axis = rotate_vectors(torch.tensor([1.0, 0.0, 0.0]), frame[3:]) * scale
    y_axis = rotate_vectors(torch.tensor([0.0, 1.0, 0.0]), frame[3:]) * scale
    z_axis = rotate_vectors(torch.tensor([0.0, 0.0, 1.0]), frame[3:]) * scale

    # Plot axes
    ax.quiver(*origin, *x_axis, color='r')
    ax.quiver(*origin, *y_axis, color='g')
    ax.quiver(*origin, *z_axis, color='b')

    # Plot origin
    ax.scatter(*origin, color=origin_color, label=label)
