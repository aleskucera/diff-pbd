import torch
import numpy as np
import matplotlib.pyplot as plt

from pbd_torch.old.body import Body
from pbd_torch.transform import rotate_vectors
from pbd_torch.animation import AnimationController


TOP_COLOR = (1.0, 0.4980392156862745, 0.054901960784313725)
BOTTOM_COLOR = (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
SURFACE_COLOR = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)

class Cylinder(Body):
    def __init__(self,
                 x: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
                 q: torch.Tensor = torch.tensor([1.0, 0.0, 0.0, 0.0]),
                 v: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
                 w: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
                 radius: float = 1.0,
                 height: float = 0.1,
                 m: float = 1.0,
                 I: torch.Tensor = None,
                 restitution: float = 0.5,
                 static_friction: float = 0.2,
                 dynamic_friction: float = 0.4,
                 n_collision_points_base: int = 10,
                 n_collision_points_surface: int = 30):
        self.radius = radius
        self.height = height
        self.n_base = n_collision_points_base
        self.n_surface = n_collision_points_surface

        super().__init__(x=x, q=q, v=v, w=w,
                         m=m, I=I, restitution=restitution,
                         static_friction=static_friction,
                         dynamic_friction=dynamic_friction)

    def _default_moment_of_inertia(self):
        I_zz = 0.5 * self.m * self.radius ** 2
        I_xx = (1 / 12) * self.m * (3 * self.radius ** 2 + self.height ** 2)
        I_yy = I_xx

        # Create 3x3 matrix for moment of inertia
        I = torch.tensor([
            [I_xx, 0.0, 0.0],
            [0.0, I_yy, 0.0],
            [0.0, 0.0, I_zz]
        ])

        return I

    def _generate_fibonacci_surface_points(self, n_points: int):
        """Generate uniformly distributed points on cylinder surface using Fibonacci spiral."""
        golden_ratio = (1 + np.sqrt(5)) / 2

        points = []
        for i in range(n_points):
            z = self.height * (i / (n_points - 1) - 0.5)
            theta = 2 * np.pi * i / golden_ratio

            x = self.radius * np.cos(theta)
            y = self.radius * np.sin(theta)
            points.append([x, y, z])

        return torch.tensor(points)

    def _generate_fibonacci_disk_points(self, n_points: int):
        """Generate uniformly distributed points on circle using sunflower pattern."""
        golden_angle = np.pi * (3 - np.sqrt(5))

        indices = torch.arange(n_points, dtype=torch.float32)
        r = self.radius * torch.sqrt(indices / n_points)
        theta = indices * golden_angle

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        return torch.stack([x, y], dim=1)

    def _create_collision_vertices(self):
        base_points = self._generate_fibonacci_disk_points(self.n_base)
        surface_vertices = self._generate_fibonacci_surface_points(self.n_surface)

        # Create top and bottom vertices for the bases
        top_vertices = torch.cat([
            base_points,
            torch.ones(len(base_points), 1) * self.height / 2
        ], dim=1)

        bottom_vertices = torch.cat([
            base_points,
            torch.ones(len(base_points), 1) * -self.height / 2
        ], dim=1)

        # Combine all vertices
        vertices = torch.cat([surface_vertices, top_vertices, bottom_vertices], dim=0).type(torch.float32)

        return vertices

    def plot_geometry(self, ax: plt.Axes, x: torch.Tensor, q: torch.Tensor):
        vertices = self.coll_vertices
        vertices_transformed = rotate_vectors(vertices, q) + x

        surface_vertices = vertices_transformed[:self.n_surface]
        top_vertices = vertices_transformed[self.n_surface:self.n_surface + self.n_base]
        bottom_vertices = vertices_transformed[self.n_surface + self.n_base:]

        # Plot collision vertices
        ax.scatter(surface_vertices[:, 0],
                   surface_vertices[:, 1],
                   surface_vertices[:, 2],
                   c=SURFACE_COLOR, marker='o', s=10,
                   alpha=0.5, label='Surface Vertices')
        ax.scatter(top_vertices[:, 0],
                   top_vertices[:, 1],
                   top_vertices[:, 2],
                   c=TOP_COLOR, marker='o', s=10,
                   alpha=0.5, label='Top Base Vertices')
        ax.scatter(bottom_vertices[:, 0],
                   bottom_vertices[:, 1],
                   bottom_vertices[:, 2],
                   c=BOTTOM_COLOR, marker='o', s=10,
                   alpha=0.5, label='Bottom Base Vertices')

        # Add wireframe for reference
        theta = torch.linspace(0, 2 * np.pi, 32)
        z = torch.linspace(-self.height / 2, self.height / 2, 8)
        theta_grid, z_grid = torch.meshgrid(theta, z, indexing='ij')
        x_grid = self.radius * torch.cos(theta_grid)
        y_grid = self.radius * torch.sin(theta_grid)

        # Transform wireframe points
        wire_points = torch.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], dim=1)
        wire_transformed = rotate_vectors(wire_points, q) + x

        # Reshape transformed points
        x_transformed = wire_transformed[:, 0].reshape(theta_grid.shape)
        y_transformed = wire_transformed[:, 1].reshape(theta_grid.shape)
        z_transformed = wire_transformed[:, 2].reshape(theta_grid.shape)

        ax.plot_wireframe(x_transformed, y_transformed, z_transformed,
                          color='gray', alpha=0.3, linewidth=1.0)
        
        # Add radial lines for top and bottom faces
        n_radial = 16   # Number of radial lines
        radial_theta = torch.linspace(0, 2 * np.pi, n_radial)

        # Create radial lines points
        for theta in radial_theta:
            # Points for a single radial line
            r_points = torch.stack([
                torch.tensor([0, self.radius * torch.cos(theta)]),  # x coordinates
                torch.tensor([0, self.radius * torch.sin(theta)]),  # y coordinates
                torch.tensor([self.height/2, self.height/2])        # z coordinates for top face
            ], dim=0)

            # Transform radial line points
            r_points_transformed = rotate_vectors(r_points.T, q) + x

            # Plot top radial line
            ax.plot(r_points_transformed[:, 0],
                    r_points_transformed[:, 1],
                    r_points_transformed[:, 2],
                    color='gray', alpha=0.3, linewidth=1.0)

            # Bottom face (just change z coordinate)
            r_points[2, :] = -self.height/2
            r_points_transformed = rotate_vectors(r_points.T, q) + x

            # Plot bottom radial line
            ax.plot(r_points_transformed[:, 0],
                    r_points_transformed[:, 1],
                    r_points_transformed[:, 2],
                    color='gray', alpha=0.3, linewidth=1.0)

def visualize_collision_model():
    radius = 1.0
    height = 0.5
    n_points_base = 50
    n_points_surface = 100
    x_lims = (-2, 2)
    y_lims = (-2, 2)
    z_lims = (-2, 2)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    cylinder = Cylinder(dt=0.0,
                        sim_time=torch.tensor([0.0]),
                        radius=radius,
                        height=height,
                        n_collision_points_base=n_points_base,
                        n_collision_points_surface=n_points_surface)

    cylinder.plot_geometry(ax, torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 0.0, 1.0]))

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
    dt = time_len / n_frames
    time = torch.linspace(0, time_len, n_frames)

    num_pos_iters = 2

    cylinder = Cylinder(x=torch.tensor([0.0, 0.0, 3.0]),
                        q=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                        v=torch.tensor([0.0, 0.0, 0.0]),
                        w=torch.tensor([1.0, 0.0, 0.0]),
                        radius=2.0,
                        height=0.75,
                        m=1.0,
                        n_collision_points_base=50,
                        n_collision_points_surface=100,
                        restitution=0.6,
                        static_friction=0.4,
                        dynamic_friction=0.5)

    cylinder.simulate(n_frames, time_len)

    controller = AnimationController(bodies=[cylinder],
                                     time=time,
                                     x_lims=(-3, 3),
                                     y_lims=(-3, 3),
                                     z_lims=(-1, 5))
    controller.start()
    
def simulate_rotation():
    n_frames = 200
    time_len = 3.0
    dt = time_len / n_frames
    time = torch.linspace(0, time_len, n_frames)

    num_pos_iters = 2

    cylinder = Cylinder(x=torch.tensor([0.0, 0.0, 2.1]),
                        q=torch.tensor([0.7071, 0.7071, 0.0, 0.0]),
                        v=torch.tensor([0.0, 0.0, 0.0]),
                        w=torch.tensor([0.0, 2.0, 0.0]),
                        radius=2.0,
                        height=0.75,
                        m=1.0,
                        n_collision_points_base=50,
                        n_collision_points_surface=100,
                        restitution=0.6,
                        static_friction=0.4,
                        dynamic_friction=1.0)

    cylinder.simulate(n_frames, time_len)

    controller = AnimationController(bodies=[cylinder],
                                     time=time,
                                     x_lims=(-3, 3),
                                     y_lims=(-3, 3),
                                     z_lims=(-1, 5))
    controller.start()

def simulate_rolling():
    n_frames = 200
    time_len = 3.0
    dt = time_len / n_frames
    time = torch.linspace(0, time_len, n_frames)

    num_pos_iters = 2

    cylinder = Cylinder(x=torch.tensor([0.0, 0.0, 2.1]),
                        q=torch.tensor([0.7071, 0.7071, 0.0, 0.0]),
                        v=torch.tensor([0.0, 0.0, 0.0]),
                        w=torch.tensor([0.0, 0.0, -3.0]),
                        radius=2.0,
                        height=0.75,
                        m=1.0,
                        n_collision_points_base=50,
                        n_collision_points_surface=100,
                        restitution=0.4,
                        static_friction=0.4,
                        dynamic_friction=1.0)

    cylinder.simulate(n_frames, time_len)

    controller = AnimationController(bodies=[cylinder],
                                     time=time,
                                     x_lims=(-1-1, 5+1),
                                     y_lims=(-3-1, 3+1),
                                     z_lims=(-1-1, 5+1))
    controller.start()

if __name__ == "__main__":
    visualize_collision_model()
    # simulate_fall()
    simulate_rotation()
    # simulate_rolling()