import os
import time

import torch
from demos.utils import save_simulation
from pbd_torch.collision import collide
from pbd_torch.newton_engine import NonSmoothNewtonEngine
from pbd_torch.constants import ROT_90_X
from pbd_torch.constants import ROT_IDENTITY
from pbd_torch.constants import ROT_NEG_90_X
from pbd_torch.model import Model
from pbd_torch.model import Quaternion
from pbd_torch.model import Vector3
from pbd_torch.terrain import create_terrain_from_exr_file
from pbd_torch.xpbd_engine import XPBDEngine
from tqdm import tqdm


def main():
    dt = 0.01
    n_steps = 300
    device = torch.device("cpu")
    collision_margin = 0.0
    dynamic_friction_threshold = 0.2
    output_file = os.path.join("simulation", "helhest_terrain_climb.json")

    terrain = create_terrain_from_exr_file(
        heightmap_path="/home/kuceral4/school/diff-pbd/data/blender/ground4/textures/ant01.002_heightmap.exr",
        size_x=40.0,
        size_y=40.0,
        device=device,
    )

    model = Model(
        terrain=terrain,
        device=device,
        dynamic_friction_threshold=dynamic_friction_threshold,
        max_contacts_per_body=16,
    )

    # Add robot base
    base = model.add_box(
        m=5.0,
        hx=1.0,
        hy=2.0,
        hz=1.0,
        name="box",
        pos=Vector3(torch.tensor([-8.0, 0.0, 3.0])),
        rot=Quaternion(ROT_IDENTITY),
        n_collision_points=500,
        restitution=0.2,
        dynamic_friction=0.8,
    )

    # Add left wheel
    left_wheel = model.add_cylinder(
        m=1.0,
        radius=2.0,
        height=0.4,
        name="left_wheel",
        pos=Vector3(torch.tensor([-8.0, 3.0, 3.0])),
        rot=Quaternion(ROT_NEG_90_X),
        n_collision_points_base=128,
        n_collision_points_surface=128,
        restitution=0.1,
        dynamic_friction=0.8,
    )

    # Add right wheel
    right_wheel = model.add_cylinder(
        m=1.0,
        radius=2.0,
        height=0.4,
        name="right_wheel",
        pos=Vector3(torch.tensor([-8.0, -3.0, 3.0])),
        rot=Quaternion(ROT_90_X),
        n_collision_points_base=128,
        n_collision_points_surface=128,
        restitution=0.1,
        dynamic_friction=0.8,
    )

    # Back wheel
    back_wheel = model.add_cylinder(
        m=1.0,
        radius=2.0,
        height=0.4,
        name="back_wheel",
        pos=Vector3(torch.tensor([-12.0, 0.0, 3.0])),
        rot=Quaternion(ROT_NEG_90_X),
        n_collision_points_base=128,
        n_collision_points_surface=128,
        restitution=0.1,
        dynamic_friction=0.8,
    )

    # # Add left hinge joint
    # left_wheel_joint = model.add_hinge_joint(
    #     parent=base,
    #     child=left_wheel,
    #     axis=Vector3(torch.tensor([0.0, 1.0, 0.0])),
    #     name="left_wheel_joint",
    #     parent_trans=torch.tensor([0.0, 2.5, 0.0, 1.0, 0.0, 0.0, 0.0]),
    #     child_trans=torch.cat((torch.tensor([0.0, 0.0, -0.5]), ROT_90_X)),
    # )

    # # Add right hinge joint
    # right_wheel_joint = model.add_hinge_joint(
    #     parent=base,
    #     child=right_wheel,
    #     axis=Vector3(torch.tensor([0.0, 1.0, 0.0])),
    #     name="right_wheel_joint",
    #     parent_trans=torch.tensor([0.0, -2.5, 0.0, 1.0, 0.0, 0.0, 0.0]),
    #     child_trans=torch.cat((torch.tensor([0.0, 0.0, -0.5]), ROT_NEG_90_X)),
    # )

    # # Add back hinge joint
    # back_wheel_joint = model.add_hinge_joint(
    #     parent=base,
    #     child=back_wheel,
    #     axis=Vector3(torch.tensor([0.0, 1.0, 0.0])),
    #     name="back_wheel_joint",
    #     parent_trans=torch.tensor([-4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    #     child_trans=torch.cat((torch.tensor([0.0, 0.0, 0.0]), ROT_90_X)),
    # )

    engine = NonSmoothNewtonEngine(model, iterations=200)

    # Set up the initial state
    states = [model.state() for _ in range(n_steps)]

    control = model.control()

    for i in tqdm(range(n_steps - 1), desc="Simulating"):
        coll_time = time.time()
        collide(model, states[i], collision_margin=collision_margin)
        # coll_time_batch = time.time()
        # collide_batch(model, states[i], collision_margin=collision_margin)
        # print(f"Collision time batch: {time.time() - coll_time_batch}")
        # control.joint_act[left_wheel_joint] = 120.0
        # control.joint_act[right_wheel_joint] = 120.0
        # control.joint_act[back_wheel_joint] = 200.0
        sim_time = time.time()
        engine.simulate(states[i], states[i + 1], control, dt)
        # print(f"Simulation time: {time.time() - sim_time}")

    print(f"Saving simulation to {output_file}")
    save_simulation(model, states, output_file)


if __name__ == "__main__":
    main()
