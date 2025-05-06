import os

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
from pbd_torch.xpbd_engine import XPBDEngine
from tqdm import tqdm


def main():
    dt = 0.001
    n_steps = 5000
    device = torch.device("cuda")
    collision_margin = 0.0
    dynamic_friction_threshold = 0.2
    output_file = os.path.join("simulation", "helhest_flat_ground.json")

    model = Model(
        device=device,
        dynamic_friction_threshold=dynamic_friction_threshold,
        max_contacts_per_body=16,
    )

    # Add robot base
    base = model.add_box(
        m=3.0,
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
        dynamic_friction=1.0,
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
        dynamic_friction=1.0,
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

    # Add left hinge joint
    left_wheel_joint = model.add_hinge_joint(
        parent=base,
        child=left_wheel,
        axis=Vector3(torch.tensor([0.0, 0.0, 1.0])),
        name="left_wheel_joint",
        parent_trans=torch.cat((torch.tensor([0.0, 2.5, 0.0]), ROT_NEG_90_X)),
        child_trans=torch.tensor([0.0, 0.0, -0.5, 1.0, 0.0, 0.0, 0.0]),
    )

    # Add right hinge joint
    right_wheel_joint = model.add_hinge_joint(
        parent=base,
        child=right_wheel,
        axis=Vector3(torch.tensor([0.0, 0.0, 1.0])),
        name="right_wheel_joint",
        parent_trans=torch.cat((torch.tensor([0.0, -2.5, 0.0]), ROT_90_X)),
        child_trans=torch.tensor([0.0, 0.0, -0.5, 1.0, 0.0, 0.0, 0.0]),
    )

    # Add back hinge joint
    back_wheel_joint = model.add_hinge_joint(
        parent=base,
        child=back_wheel,
        axis=Vector3(torch.tensor([0.0, 0.0, 1.0])),
        name="back_wheel_joint",
        parent_trans=torch.cat((torch.tensor([-4.0, 0.0, 0.0]), ROT_NEG_90_X)),
        child_trans=torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )

    engine = NonSmoothNewtonEngine(model, iterations=100, device=device)
    xpbd_engine = XPBDEngine(model)

    # Set up the initial state
    states = [model.state() for _ in range(n_steps)]

    control = model.control()

    for i in tqdm(range(n_steps - 1), desc="Simulating"):
        collide(model, states[i], collision_margin=collision_margin)

        control.add_actuation(left_wheel_joint, 15)
        control.add_actuation(right_wheel_joint, -15)

        xpbd_engine.simulate(states[i], states[i + 1], control, dt)

    print(f"Saving simulation to {output_file}")
    save_simulation(model, states, output_file)


if __name__ == "__main__":
    main()
