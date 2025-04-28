import os

import torch
from demos.utils import save_simulation
from pbd_torch.collision import collide, collide_terrain_friction
from pbd_torch.newton_engine_friction import NonSmoothNewtonEngine
from pbd_torch.constants import ROT_90_X
from pbd_torch.constants import ROT_IDENTITY
from pbd_torch.constants import ROT_NEG_90_X
from pbd_torch.model import Model
from pbd_torch.model import Quaternion
from pbd_torch.model import Vector3
from pbd_torch.terrain import create_terrain_from_exr_file
from tqdm import tqdm

def main():
    dt = 0.01
    n_steps = 300
    device = torch.device("cpu")
    collision_margin = 0.0
    friction_collision_margin = 0.08
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
        n_collision_points_base=256,
        n_collision_points_surface=256,
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
        n_collision_points_base=256,
        n_collision_points_surface=256,
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
        n_collision_points_base=256,
        n_collision_points_surface=256,
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
        ke=50.0,
        kd=0.5,
    )

    # Add right hinge joint
    right_wheel_joint = model.add_hinge_joint(
        parent=base,
        child=right_wheel,
        axis=Vector3(torch.tensor([0.0, 1.0, 0.0])),
        name="right_wheel_joint",
        parent_trans=torch.cat((torch.tensor([0.0, -2.5, 0.0]), ROT_90_X)),
        child_trans=torch.tensor([0.0, 0.0, -0.5, 1.0, 0.0, 0.0, 0.0]),
        ke=50.0,
        kd=0.5,
    )

    # Add back hinge joint
    back_wheel_joint = model.add_hinge_joint(
        parent=base,
        child=back_wheel,
        axis=Vector3(torch.tensor([0.0, 1.0, 0.0])),
        name="back_wheel_joint",
        parent_trans=torch.cat((torch.tensor([-4.0, 0.0, 0.0]), ROT_NEG_90_X)),
        child_trans=torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ke=0.5,
        kd=0.5,
    )

    # Create engine from model
    engine = NonSmoothNewtonEngine(model, iterations=100)

    control = model.control()
    states = [model.state() for _ in range(n_steps)]

    for i in tqdm(range(n_steps - 1), desc="Simulating"):
        collide(model, states[i], collision_margin=collision_margin)
        collide_terrain_friction(model, states[i], collision_margin=friction_collision_margin)

        control.add_actuation(left_wheel_joint, 3)
        control.add_actuation(right_wheel_joint, -3)

        engine.simulate_xitorch(states[i], states[i + 1], control, dt)

    print(f"Saving simulation to {output_file}")
    save_simulation(model, states, output_file)


if __name__ == "__main__":
    main()
