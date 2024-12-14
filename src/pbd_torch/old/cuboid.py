import torch
import matplotlib.pyplot as plt

from pbd_torch.old.body import Body
from pbd_torch.transform import rotate_vectors
from pbd_torch.animation import AnimationController

TOP_COLOR = (1.0, 0.4980392156862745, 0.054901960784313725)
BOTTOM_COLOR = (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
SURFACE_COLOR = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)


class Cuboid(Body):
    def __init__(self,
                 x: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
                 q: torch.Tensor = torch.tensor([1.0, 0.0, 0.0, 0.0]),
                 v: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
                 w: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
                 hx: float = 0.5,
                 hy: float = 0.5,
                 hz: float = 0.5,
                 m: float = 1.0,
                 I: torch.Tensor = None,
                 restitution: float = 0.5,
                 static_friction: float = 0.2,
                 dynamic_friction: float = 0.4,
                 n_collision_points: int = 10):
        self.hx = hx
        self.hy = hy
        self.hz = hz
        self.n_collision_points = n_collision_points

        super().__init__(x=x, q=q, v=v, w=w,
                         m=m, I=I, restitution=restitution,
                         static_friction=static_friction,
                         dynamic_friction=dynamic_friction)

    def _default_moment_of_inertia(self):
        I = torch.tensor([
            [1 / 12 * self.m * (3 * self.hy ** 2 + self.hz ** 2), 0, 0],
            [0, 1 / 12 * self.m * (3 * self.hx ** 2 + self.hz ** 2), 0],
            [0, 0, 1 / 12 * self.m * (3 * self.hx ** 2 + self.hy ** 2)]
        ])
        return I

    @staticmethod
    def _create_face_points(x_coords, y_coords, fixed_dim, fixed_val):
        XX, YY = torch.meshgrid(x_coords, y_coords, indexing='ij')
        points = torch.stack([XX.flatten(), YY.flatten()], dim=1)

        # Create the full 3D points based on which dimension is fixed
        if fixed_dim == 0:  # YZ plane (fixed X)
            vertices = torch.column_stack([
                torch.full_like(points[:, 0], fixed_val),
                points[:, 0],
                points[:, 1]
            ])
        elif fixed_dim == 1:  # XZ plane (fixed Y)
            vertices = torch.column_stack([
                points[:, 0],
                torch.full_like(points[:, 0], fixed_val),
                points[:, 1]
            ])
        else:  # XY plane (fixed Z)
            vertices = torch.column_stack([
                points[:, 0],
                points[:, 1],
                torch.full_like(points[:, 0], fixed_val)
            ])

        return vertices

    def plot_geometry(self, ax: plt.Axes, x: torch.Tensor, q: torch.Tensor):
        points = rotate_vectors(self.coll_vertices, q) + x

        # Plot the collision points
        ax.scatter(points[:, 0],
                   points[:, 1],
                   points[:, 2],
                   c=self.coll_vertices[:, 2], marker='o',
                   s=10, cmap='viridis', alpha=0.7)

        corners = torch.tensor([[-self.hx, -self.hy, -self.hz],
                                [self.hx, -self.hy, -self.hz],
                                [self.hx, self.hy, -self.hz],
                                [-self.hx, self.hy, -self.hz],
                                [-self.hx, -self.hy, self.hz],
                                [self.hx, -self.hy, self.hz],
                                [self.hx, self.hy, self.hz],
                                [-self.hx, self.hy, self.hz]])

        rotated_corners = rotate_vectors(corners, q) + x
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        for edge in edges:
            start, end = rotated_corners[edge[0]], rotated_corners[edge[1]]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='gray', alpha=0.5, linewidth=1)

    def _create_collision_vertices(self):

        # Split the number of points based on the surface area (proportional to the number of points)
        n_points = int(self.n_collision_points / 6)

        # Create linearly spaced points along each dimension
        x = torch.linspace(-self.hx, self.hx, n_points)
        y = torch.linspace(-self.hy, self.hy, n_points)
        z = torch.linspace(-self.hz, self.hz, n_points)

        vertices_list = []
        vertices_list.append(self._create_face_points(y, z, 0, self.hx))
        vertices_list.append(self._create_face_points(y, z, 0, -self.hx))
        vertices_list.append(self._create_face_points(x, z, 1, self.hy))
        vertices_list.append(self._create_face_points(x, z, 1, -self.hy))
        vertices_list.append(self._create_face_points(x, y, 2, self.hz))
        vertices_list.append(self._create_face_points(x, y, 2, -self.hz))

        vertices = torch.cat(vertices_list, dim=0)

        return vertices


def visualize_collision_model():
    n_points = 40
    x_lims = (-2, 2)
    y_lims = (-2, 2)
    z_lims = (-2, 2)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    cuboid = Cuboid(hx=0.5,
                    hy=0.5,
                    hz=0.5,
                    n_collision_points=n_points)

    cuboid.plot_geometry(ax, torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 0.0, 1.0]))

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

    cuboid = Cuboid(x=torch.tensor([0.0, 0.0, 3.0]),
                    q=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    v=torch.tensor([0.0, 0.0, 0.0]),
                    w=torch.tensor([0.0, 1.0, 1.0]),
                    hx=1.0,
                    hy=1.0,
                    hz=1.0,
                    m=1.0,
                    n_collision_points=32)

    cuboid.simulate(n_frames, time_len)

    # for i in range(n_frames):
    #     cuboid.integrate()
    #     cuboid.detect_collisions()
    #     for j in range(num_pos_iters):
    #         cuboid.correct_collisions()
    #     cuboid.update_velocity()
    #     cuboid.solve_velocity()
    #     cuboid.save_state(i)
    #     cuboid.save_collisions(i)

    controller = AnimationController(bodies=[cuboid],
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

    cuboid = Cuboid(x=torch.tensor([0.0, 0.0, 1.1]),
                    q=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    v=torch.tensor([0.0, 0.0, 0.0]),
                    w=torch.tensor([0.0, 0.0, 3.0]),
                    hx=1.0,
                    hy=1.0,
                    hz=1.0,
                    m=1.0,
                    n_collision_points=32)

    cuboid.simulate(n_frames, time_len)

    # for i in range(n_frames):
    #     cuboid.integrate()
    #     cuboid.detect_collisions()
    #     for j in range(num_pos_iters):
    #         cuboid.correct_collisions()
    #     cuboid.update_velocity()
    #     cuboid.solve_velocity()
    #     cuboid.save_state(i)
    #     cuboid.save_collisions(i)

    controller = AnimationController(bodies=[cuboid],
                                     time=time,
                                     x_lims=(-3, 3),
                                     y_lims=(-3, 3),
                                     z_lims=(-1, 5))
    controller.start()

if __name__ == "__main__":
    visualize_collision_model()
    simulate_fall()
    simulate_rotation()