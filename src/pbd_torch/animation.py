from typing import List
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from pbd_torch.old.body import Body
from pbd_torch.constants import ROT_IDENTITY
from pbd_torch.integrator import XPBDIntegrator
from pbd_torch.model import Model
from pbd_torch.model import State
from pbd_torch.transform import rotate_vectors

matplotlib.use('TkAgg')

MAX_COLLISIONS = 30


class AnimationController:

    def __init__(self,
                 bodies: List[Body],
                 time: torch.Tensor,
                 x_lims: Tuple[float, float] = (-4, 4),
                 y_lims: Tuple[float, float] = (-4, 4),
                 z_lims: Tuple[float, float] = (-4, 4)):
        self.time = time
        self.bodies = bodies

        self.current_frame = 0
        self.current_body_index = None  # None means show all boxes

        self.x_lims = x_lims
        self.y_lims = y_lims
        self.z_lims = z_lims

        # Set up the plot
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.update_plot()  # Display the first frame

    def on_key_press(self, event):
        if event.key == 'n':  # Next frame
            self.current_frame = (self.current_frame + 1) % len(self.time)
            self.update_plot()

        elif event.key == 'N':  # Jump forward by 10 frames
            self.current_frame = (self.current_frame + 10) % len(self.time)
            self.update_plot()

        elif event.key == 'b':  # Previous frame
            self.current_frame = (self.current_frame - 1) % len(self.time)
            self.update_plot()

        elif event.key == 'B':  # Jump backward by 10 frames
            self.current_frame = (self.current_frame - 10) % len(self.time)
            self.update_plot()

        elif event.key in map(str, range(len(
                self.bodies))):  # Visualize a specific box
            self.current_body_index = int(event.key)
            self.update_plot()

        elif event.key == 'a':  # Visualize all boxes
            self.current_body_index = None
            self.update_plot()

    def update_plot(self):
        self.ax.cla()
        self.ax.set_title(f"Time: {self.time[self.current_frame]:.2f}")
        self.ax.set_xlim(*self.x_lims)
        self.ax.set_ylim(*self.y_lims)
        self.ax.set_zlim(*self.z_lims)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        if self.current_body_index is None:  # Show all boxes
            for box in self.bodies:
                box.plot(self.ax, self.current_frame)
        else:  # Show only the selected box
            self.bodies[self.current_body_index].plot(self.ax,
                                                      self.current_frame)

        # Make the axis equal
        self.ax.set_box_aspect([1, 1, 1])

        plt.draw()

    @staticmethod
    def start():
        plt.show()


class AnimationController2:

    def __init__(self,
                 model: Model,
                 states: List[State],
                 x_lims: Tuple[float, float] = (-4, 4),
                 y_lims: Tuple[float, float] = (-4, 4),
                 z_lims: Tuple[float, float] = (-4, 4)):
        self.model = model
        self.states = states

        self.current_frame = 0

        self.x_lims = x_lims
        self.y_lims = y_lims
        self.z_lims = z_lims

        # Set up the plot
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_key_press(self, event):
        if event.key == 'n':  # Next frame
            self.current_frame = (self.current_frame + 1) % len(self.states)
            self.update_plot()

        elif event.key == 'N':  # Jump forward by 10 frames
            self.current_frame = (self.current_frame + 10) % len(self.states)
            self.update_plot()

        elif event.key == 'b':  # Previous frame
            self.current_frame = (self.current_frame - 1) % len(self.states)
            self.update_plot()

        elif event.key == 'B':  # Jump backward by 10 frames
            self.current_frame = (self.current_frame - 10) % len(self.states)
            self.update_plot()

    def update_plot(self):
        self.ax.cla()
        self.ax.set_xlim(*self.x_lims)
        self.ax.set_ylim(*self.y_lims)
        self.ax.set_zlim(*self.z_lims)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        state = self.states[self.current_frame]

        for b in range(self.model.body_count):
            collision_points = self.model.body_collision_points[b]
            body_q = state.body_q[b]
            plot_collision_points(self.ax, collision_points, body_q)

        # Make the axis equal
        self.ax.set_box_aspect([1, 1, 1])

        plt.draw()

    def start(self):
        plt.show()


def plot_collision_points(ax: Axes, points: torch.Tensor,
                          body_q: torch.Tensor):
    color = points[:, 2].clone()
    points = rotate_vectors(points.clone(), body_q[3:]) + body_q[:3]

    ax.scatter(points[:, 0],
               points[:, 1],
               points[:, 2],
               c=color,
               marker='o',
               s=10,
               cmap='viridis',
               alpha=0.7)


def animation_demo():
    dt = 0.01
    n_steps = 100

    model = Model()
    integrator = XPBDIntegrator()

    # Add a body to the model
    model.add_sphere(m=1.0,
                     radius=0.5,
                     name='sphere',
                     pos=torch.tensor([0.0, 0.0, 1.0]),
                     rot=ROT_IDENTITY,
                     n_collision_points=100)

    # Set up the initial state
    states = [model.state() for _ in range(n_steps)]

    # Simulati the model
    for i in range(n_steps - 1):
        integrator.simulate(model, states[i], states[i + 1], dt)
        print(f"Step {i}: {states[i].body_q[0]}")

    animator = AnimationController2(model, states)

    animator.start()


if __name__ == "__main__":
    animation_demo()
