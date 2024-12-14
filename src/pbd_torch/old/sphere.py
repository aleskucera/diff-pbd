import torch
import numpy as np
import matplotlib.pyplot as plt

from pbd_torch.old.body import Body
from pbd_torch.transform import rotate_vectors
from pbd_torch.animation import AnimationController


class Sphere(Body):
    def __init__(self,
                 x: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
                 q: torch.Tensor = torch.tensor([1.0, 0.0, 0.0, 0.0]),
                 v: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
                 w: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
                 radius: float = 1.0,
                 m: float = 1.0,
                 I: torch.Tensor = None,
                 restitution: float = 0.5,
                 static_friction: float = 0.2,
                 dynamic_friction: float = 0.4,
                 n_collision_points: int = 100,
                 render_color: tuple = (0.0, 1.0, 0.0)):
        self.radius = radius
        self.n_collision_points = n_collision_points

        super().__init__(x=x, q=q, v=v, w=w,
                         m=m, I=I, restitution=restitution,
                         static_friction=static_friction,
                         dynamic_friction=dynamic_friction,
                         render_color=render_color)

    def _default_moment_of_inertia(self):
        I = 0.4 * self.m * self.radius ** 2 * torch.eye(3)
        return I

    def _generate_fibonacci_sphere_points(self, n_points: int):
        indices = torch.arange(n_points, dtype=torch.float)

        # Golden ratio constant
        phi = (torch.sqrt(torch.tensor(5.0)) - 1) / 2

        # Calculate z coordinates
        z = (2 * indices + 1) / n_points - 1

        # Calculate radius at each z
        radius_at_z = torch.sqrt(1 - z * z)

        # Calculate angles
        theta = 2 * torch.pi * indices * phi

        # Calculate x, y, z coordinates
        x = self.radius * radius_at_z * torch.cos(theta)
        y = self.radius * radius_at_z * torch.sin(theta)
        z = self.radius * z

        # Stack coordinates
        points = torch.stack([x, y, z], dim=1)

        return points

    def _create_collision_vertices(self):
        return self._generate_fibonacci_sphere_points(self.n_collision_points)

    def plot_geometry(self, ax: plt.Axes, x: torch.Tensor, q: torch.Tensor):
        points = rotate_vectors(self.coll_vertices, q) + x

        # Plot the collision points
        ax.scatter(points[:, 0],
                   points[:, 1],
                   points[:, 2],
                   c=self.coll_vertices[:, 2], marker='o',
                   s=10, cmap='viridis', alpha=0.7)

        # Add wireframe for reference
        u = torch.linspace(0, 2 * np.pi, 30)  # Longitude angles
        v = torch.linspace(0, np.pi, 30)  # Latitude angles

        X = self.radius * torch.outer(torch.cos(u), torch.sin(v))  # X-coordinates
        Y = self.radius * torch.outer(torch.sin(u), torch.sin(v))  # Y-coordinates
        Z = self.radius * torch.outer(torch.ones_like(u), torch.cos(v))  # Z-coordinates

        points = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], dim=1)
        points_transformed = rotate_vectors(points, q) + x
        X, Y, Z = points_transformed[:, 0], points_transformed[:, 1], points_transformed[:, 2]
        X = X.reshape(u.shape[0], v.shape[0])
        Y = Y.reshape(u.shape[0], v.shape[0])
        Z = Z.reshape(u.shape[0], v.shape[0])

        # Plot the wireframe
        ax.plot_wireframe(X, Y, Z, color='gray', linewidth=0.5, alpha=0.3)


def visualize_collision_model():
    radius = 1.0
    n_points = 200
    x_lims = (-2, 2)
    y_lims = (-2, 2)
    z_lims = (-2, 2)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    sphere = Sphere(radius=radius,
                    n_collision_points=n_points)

    sphere.plot_geometry(ax, torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 0.0, 1.0]))

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_zlim(z_lims)
    ax.set_title('Cylinder Wireframe')

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    plt.show()


def simulate_fall():
    # Create sample trajectory data
    n_frames = 200
    time_len = 3.0
    time = torch.linspace(0, time_len, n_frames)

    sphere = Sphere(x=torch.tensor([0.0, 0.0, 3.0]),
                    q=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    v=torch.tensor([0.0, 0.0, 0.0]),
                    w=torch.tensor([1.0, 0.0, 0.0]),
                    radius=2.0,
                    m=1.0,
                    n_collision_points=150,
                    restitution=0.8,
                    static_friction=0.4,
                    dynamic_friction=1.0)

    sphere.simulate(n_frames, time_len)

    controller = AnimationController(bodies=[sphere],
                                     time=time,
                                     x_lims=(-3, 3),
                                     y_lims=(-3, 3),
                                     z_lims=(-1, 5))
    controller.start()


def simulate_rotation():
    n_frames = 200
    time_len = 3.0
    time = torch.linspace(0, time_len, n_frames)

    sphere = Sphere(x=torch.tensor([0.0, 0.0, 2.1]),
                    q=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    v=torch.tensor([0.0, 0.0, 0.0]),
                    w=torch.tensor([0.0, 0.0, 3.0]),
                    radius=2.0,
                    m=1.0,
                    n_collision_points=300,
                    restitution=0.8,
                    static_friction=0.4,
                    dynamic_friction=1.0)

    sphere.simulate(n_frames, time_len)
    controller = AnimationController(bodies=[sphere],
                                     time=time,
                                     x_lims=(-3, 3),
                                     y_lims=(-3, 3),
                                     z_lims=(-1, 5))
    controller.start()


def simulate_roll():
    # Create sample trajectory data
    n_frames = 200
    time_len = 3.0
    time = torch.linspace(0, time_len, n_frames)

    sphere = Sphere(x=torch.tensor([0.0, 0.0, 2.1]),
                    q=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    v=torch.tensor([0.0, 0.0, 0.0]),
                    w=torch.tensor([0.0, 5.0, 0.0]),
                    radius=2.0,
                    m=1.0,
                    n_collision_points=150,
                    restitution=0.4,
                    static_friction=0.4,
                    dynamic_friction=1.0)

    sphere.simulate(n_frames, time_len)
    controller = AnimationController(bodies=[sphere],
                                     time=time,
                                     x_lims=(-2, 6),
                                     y_lims=(-4, 4),
                                     z_lims=(-2, 6))
    controller.start()


if __name__ == "__main__":
    visualize_collision_model()
    simulate_fall()
    simulate_rotation()
    simulate_roll()
