import matplotlib
import matplotlib.pyplot as plt
import torch
from demos.utils import plot_frame
from demos.utils import plot_points
from pbd_torch.constants import ROT_90_X
from pbd_torch.constants import ROT_IDENTITY
from pbd_torch.constants import ROT_NEG_90_X
from pbd_torch.model import Model
from pbd_torch.model import Quaternion
from pbd_torch.model import Vector3
from pbd_torch.transform import transform_multiply
from pbd_torch.transform import transform_points_batch

matplotlib.use("TkAgg")


def show_joint():
    model = Model()
    base = model.add_box(
        m=10.0,
        hx=1.0,
        hy=1.0,
        hz=1.0,
        name="box",
        pos=Vector3(torch.tensor([0.0, 0.0, 2.1])),
        rot=Quaternion(ROT_IDENTITY),
        n_collision_points=200,
        restitution=0.1,
        dynamic_friction=1.0,
    )
    wheel = model.add_cylinder(
        m=1.0,
        radius=2.0,
        height=0.4,
        name="left_wheel",
        pos=Vector3(torch.tensor([0.0, 2.5, 2.1])),
        rot=Quaternion(ROT_NEG_90_X),
        n_collision_points_base=64,
        n_collision_points_surface=64,
        restitution=0.1,
        dynamic_friction=1.0,
    )

    # Add left hinge joint
    joint = model.add_hinge_joint(
        parent=base,
        child=wheel,
        axis=Vector3(torch.tensor([0.0, 1.0, 0.0])),
        name="left_wheel_joint",
        parent_xform=torch.tensor([0.0, 1.5, 0.0, 1.0, 0.0, 0.0, 0.0]),
        child_xform=torch.cat((torch.tensor([0.0, 0.0, -1.0]), ROT_90_X)),
    )

    base_q = model.body_q[base]
    wheel_q = model.body_q[wheel]
    base_points = model.body_collision_points[base]
    wheel_points = model.body_collision_points[wheel]
    base_points_world = transform_points_batch(
        base_points, base_q.expand(base_points.shape[0], -1)
    )
    wheel_points_world = transform_points_batch(
        wheel_points, wheel_q.expand(wheel_points.shape[0], -1)
    )

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the collision points of the bodies
    plot_points(ax, base_points_world, "Base Body")
    plot_points(ax, wheel_points_world, "Wheel Body")

    # Plot the body frames (base is larger than wheel)
    plot_frame(ax, base_q, "r", scale=1.0, label="Base Frame")
    plot_frame(ax, wheel_q, "b", scale=0.5, label="Wheel Frame")

    # Get the joint parent and child (indices of the bodies)
    parent = model.joint_parent[joint]
    child = model.joint_child[joint]

    # Get the joint transforms
    X_pj = model.joint_X_p[joint]
    X_cj = model.joint_X_c[joint]

    # Get the world transforms of the parent and child bodies
    X_wj_p = transform_multiply(model.body_q[parent], X_pj)
    X_wj_c = transform_multiply(model.body_q[child], X_cj)

    # These transforms represent the same joint frame, so they should be equal
    print(f"X_wj_p: {X_wj_p}")
    print(f"X_wj_c: {X_wj_c}")

    # Plot the joint frame (parent is larger than child)
    plot_frame(ax, X_wj_p, "orange", scale=1.0, label="Joint Frame from Base")
    plot_frame(ax, X_wj_c, "green", scale=0.5, label="Joint Frame from Wheel")

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 6)
    ax.set_zlim(-3, 7)
    ax.legend()

    plt.show()


if __name__ == "__main__":
    show_joint()
