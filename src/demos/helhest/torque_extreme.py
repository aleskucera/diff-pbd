import os

import torch
from demos.utils import save_simulation
from pbd_torch.collision import collide
from pbd_torch.constants import ROT_90_X
from pbd_torch.constants import ROT_IDENTITY
from pbd_torch.constants import ROT_NEG_90_X
from pbd_torch.integrator import XPBDIntegrator
from pbd_torch.model import Model
from pbd_torch.model import Quaternion
from pbd_torch.model import Vector3
from pbd_torch.terrain import create_terrain_from_exr_file
from tqdm import tqdm


def main():
    dt = 0.01
    n_steps = 300
    output_file = os.path.join("simulation", "helhest_torque_extreme.json")

    terrain = create_terrain_from_exr_file(
        heightmap_path="/home/kuceral4/school/diff-pbd/data/blender/ground4/textures/ant01_heightmap.exr",
        size_x=40.0,
        size_y=40.0,
    )

    model = Model(terrain=terrain)

    integrator = XPBDIntegrator(iterations=3)

    # Add robot base
    base = model.add_box(
        m=10.0,
        hx=1.0,
        hy=2.0,
        hz=1.0,
        name="box",
        pos=Vector3(torch.tensor([0.0, 0.0, 6.4])),
        rot=Quaternion(ROT_IDENTITY),
        n_collision_points=500,
        restitution=0.1,
        dynamic_friction=1.0,
    )

    # Add left wheel
    left_wheel = model.add_cylinder(
        m=1.0,
        radius=2.0,
        height=0.4,
        name="left_wheel",
        pos=Vector3(torch.tensor([0.0, 3.0, 6.4])),
        rot=Quaternion(ROT_NEG_90_X),
        n_collision_points_base=128,
        n_collision_points_surface=128,
        restitution=0.1,
        dynamic_friction=1.0,
    )

    # Add right wheel
    right_wheel = model.add_cylinder(
        m=1.0,
        radius=2.0,
        height=0.4,
        name="right_wheel",
        pos=Vector3(torch.tensor([0.0, -3.0, 6.4])),
        rot=Quaternion(ROT_90_X),
        n_collision_points_base=128,
        n_collision_points_surface=128,
        restitution=0.1,
        dynamic_friction=1.0,
    )

    # Back wheel
    back_wheel = model.add_cylinder(
        m=1.0,
        radius=2.0,
        height=0.4,
        name="back_wheel",
        pos=Vector3(torch.tensor([-4.0, 0.0, 6.4])),
        rot=Quaternion(ROT_NEG_90_X),
        n_collision_points_base=128,
        n_collision_points_surface=128,
    )

    # Add left hinge joint
    left_wheel_joint = model.add_hinge_joint(
        parent=base,
        child=left_wheel,
        axis=Vector3(torch.tensor([0.0, 1.0, 0.0])),
        name="left_wheel_joint",
        parent_xform=torch.tensor([0.0, 2.5, 0.0, 1.0, 0.0, 0.0, 0.0]),
        child_xform=torch.cat((torch.tensor([0.0, 0.0, -0.5]), ROT_90_X)),
    )

    # Add right hinge joint
    right_wheel_joint = model.add_hinge_joint(
        parent=base,
        child=right_wheel,
        axis=Vector3(torch.tensor([0.0, 1.0, 0.0])),
        name="right_wheel_joint",
        parent_xform=torch.tensor([0.0, -2.5, 0.0, 1.0, 0.0, 0.0, 0.0]),
        child_xform=torch.cat((torch.tensor([0.0, 0.0, -0.5]), ROT_NEG_90_X)),
    )

    # Add back hinge joint
    back_wheel_joint = model.add_hinge_joint(
        parent=base,
        child=back_wheel,
        axis=Vector3(torch.tensor([0.0, 1.0, 0.0])),
        name="back_wheel_joint",
        parent_xform=torch.tensor([-4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        child_xform=torch.cat((torch.tensor([0.0, 0.0, 0.0]), ROT_90_X)),
    )

    # Set up the initial state
    states = [model.state() for _ in range(n_steps)]

    control = model.control()

    for i in tqdm(range(n_steps - 1), desc="Simulating"):
        collide(model, states[i], collision_margin=0.1)
        control.joint_act[left_wheel_joint] = -50.0
        control.joint_act[right_wheel_joint] = -50.0
        integrator.simulate(model, states[i], states[i + 1], control, dt)

    print(f"Saving simulation to {output_file}")
    save_simulation(model, states, output_file)


if __name__ == "__main__":
    main()
