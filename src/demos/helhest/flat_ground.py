import os
import time

import torch
from demos.utils import save_simulation
from pbd_torch.collision import collide_batch
from pbd_torch.constants import ROT_90_X
from pbd_torch.constants import ROT_IDENTITY
from pbd_torch.constants import ROT_NEG_90_X
from pbd_torch.model import Model
from pbd_torch.model import Quaternion
from pbd_torch.model import Vector3
from pbd_torch.xpbd_engine import XPBDEngine
from tqdm import tqdm
from pbd_torch.newton_engine import NonSmoothNewtonEngine


def main():
    dt = 0.01
    n_steps = 400
    device = torch.device("cpu")
    collision_margin = 0.2
    dynamic_friction_threshold = 0.2
    output_file = os.path.join("simulation", "helhest_flat_ground.json")

    model = Model(
        device=device,
        dynamic_friction_threshold=dynamic_friction_threshold,
        max_contacts_per_body=64,
    )

    # engine = XPBDEngine(iterations=2)
    engine = NonSmoothNewtonEngine(iterations=10, device=device)

    # Add robot base
    base = model.add_box(
        m=10.0,
        hx=1.0,
        hy=2.0,
        hz=1.0,
        name="box",
        pos=Vector3(torch.tensor([-8.0, 0.0, 2.0])),
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
        pos=Vector3(torch.tensor([-8.0, 3.0, 2.0])),
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
        pos=Vector3(torch.tensor([-8.0, -3.0, 2.0])),
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
        pos=Vector3(torch.tensor([-12.0, 0.0, 2.0])),
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
        coll_time = time.time()
        collide_batch(model, states[i], collision_margin=collision_margin)
        # print(f"Collision time: {time.time() - coll_time}")
        # control.joint_act[left_wheel_joint] = 120.0
        # control.joint_act[right_wheel_joint] = 20.0
        sim_time = time.time()
        engine.simulate(model, states[i], states[i + 1], control, dt)
        # print(f"Simulation time: {time.time() - sim_time}")

    print(f"Saving simulation to {output_file}")
    save_simulation(model, states, output_file)


if __name__ == "__main__":
    main()
