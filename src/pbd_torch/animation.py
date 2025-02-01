from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch
from pbd_torch.transform import rotate_vectors

matplotlib.use('TkAgg')

MAX_COLLISIONS = 30


@dataclass
class BodyFrame:
    transform: torch.Tensor  # [x, y, z, qw, qx, qy, qz]
    scale: float = 1.0
    color: str = 'black'
    label: str = None

    def __post_init__(self):
        # Clone the transform when the object is created
        self.transform = self.transform.clone()


class AnimationController:

    def __init__(self,
                 frames_sequence: List[Dict[str, BodyFrame]],
                 time: torch.Tensor = None,
                 x_lims: Tuple[float, float] = (-4, 4),
                 y_lims: Tuple[float, float] = (-4, 4),
                 z_lims: Tuple[float, float] = (-4, 4)):
        """
        Args:
            frames_sequence: List of dictionaries, where each dictionary contains named frames for that timestep
            time: Optional tensor of timestamps for each frame
            x_lims, y_lims, z_lims: Plot limits
        """
        self._print_help()
        self.frames_sequence = frames_sequence
        self.time = time if time is not None else torch.arange(
            len(frames_sequence))

        self.current_frame = 0
        self.current_frame_name = None  # None means show all frames

        self.x_lims = x_lims
        self.y_lims = y_lims
        self.z_lims = z_lims

        # Set up the plot
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.update_plot()

    def _print_help(self):
        print('Animation Controls:')
        print('  n: Next frame')
        print('  N: Jump forward by 10 frames')
        print('  b: Previous frame')
        print('  B: Jump backward by 10')

    def plot_frame(self, frame: BodyFrame):
        """Plot a single frame with its coordinate axes"""
        position = frame.transform[:3]
        orientation = frame.transform[3:]

        # Define axes
        x_axis = rotate_vectors(torch.tensor([1.0, 0.0, 0.0]),
                                orientation) * frame.scale
        y_axis = rotate_vectors(torch.tensor([0.0, 1.0, 0.0]),
                                orientation) * frame.scale
        z_axis = rotate_vectors(torch.tensor([0.0, 0.0, 1.0]),
                                orientation) * frame.scale

        # Plot axes
        self.ax.quiver(*position, *x_axis, color='r')
        self.ax.quiver(*position, *y_axis, color='g')
        self.ax.quiver(*position, *z_axis, color='b')

        # Plot origin
        self.ax.scatter(*position, color=frame.color, label=frame.label)

    def on_key_press(self, event):
        if event.key == 'n':  # Next frame
            self.current_frame = (self.current_frame + 1) % len(
                self.frames_sequence)
            self.update_plot()

        elif event.key == 'N':  # Jump forward by 10 frames
            self.current_frame = (self.current_frame + 10) % len(
                self.frames_sequence)
            self.update_plot()

        elif event.key == 'b':  # Previous frame
            self.current_frame = (self.current_frame - 1) % len(
                self.frames_sequence)
            self.update_plot()

        elif event.key == 'B':  # Jump backward by 10 frames
            self.current_frame = (self.current_frame - 10) % len(
                self.frames_sequence)
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

        current_frames = self.frames_sequence[self.current_frame]

        for frame in current_frames.values():
            self.plot_frame(frame)

        # Make the axis equal
        self.ax.set_box_aspect([1, 1, 1])

        if any(frame.label for frame in current_frames.values()):
            self.ax.legend()

        plt.draw()

    @staticmethod
    def start():
        plt.show()
