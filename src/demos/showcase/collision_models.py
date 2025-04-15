import matplotlib.pyplot as plt
from demos.utils import plot_points
from pbd_torch.model import box_collision_points
from pbd_torch.model import cylinder_collision_points
from pbd_torch.model import sphere_collision_points


def show_collision_models():
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    # Test sphere
    sphere_vertices = sphere_collision_points(radius=1.0, n_points=1000)
    plot_points(ax1, sphere_vertices, "Sphere Points")
    ax1.set_box_aspect([1, 1, 1])

    # Test cylinder
    cylinder_vertices = cylinder_collision_points(radius=1.0,
                                                  height=2.0,
                                                  n_base_points=500,
                                                  n_surface_points=1000)
    plot_points(ax2, cylinder_vertices, "Cylinder Points")
    ax2.set_box_aspect([1, 1, 1])

    # Test box
    box_vertices = box_collision_points(hx=2.0, hy=1.5, hz=1.0, n_points=2000)
    plot_points(ax3, box_vertices, "Box Points")
    ax3.set_box_aspect([1, 1, 1])

    plt.show()


if __name__ == "__main__":
    show_collision_models()
