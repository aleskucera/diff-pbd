import os

import torch
from demos.utils import save_simulation
from pbd_torch.collision import collide
from pbd_torch.constants import ROT_IDENTITY
from pbd_torch.model import Model
from pbd_torch.model import Quaternion
from pbd_torch.model import Vector3
from pbd_torch.newton_engine import NonSmoothNewtonEngine
from pbd_torch.terrain import create_terrain_from_exr_file
from tqdm import tqdm


def main():
    dt = 0.01
    n_steps = 400
    device = torch.device("cpu")
    output_file = os.path.join("simulation", "box.json")

    terrain = create_terrain_from_exr_file(
        heightmap_path="/home/kuceral4/school/diff-pbd/data/blender/ground4/textures/ant01.002_heightmap.exr",
        size_x=40.0,
        size_y=40.0,
        device=device,
    )

    model = Model(terrain=terrain, max_contacts_per_body=36)

    # Add robot base
    box = model.add_box(
        m=1.0,
        hx=0.5,
        hy=0.5,
        hz=0.5,
        name="box1",
        pos=Vector3(torch.tensor([-0.5, -1.0, 4.5])),
        rot=Quaternion(ROT_IDENTITY),
        restitution=0.2,
        dynamic_friction=0.8,
        n_collision_points=200,
    )

    # # Add initial rotation to the box
    # model.body_qd[box, :3] = torch.tensor([0.0, 0.0, 0.0]).view(3, 1)
    # model.body_qd[box, 3:] = torch.tensor([0.0, 0.0, 0.0]).view(3, 1)

    engine = NonSmoothNewtonEngine(model, iterations=100)

    control = model.control()
    states = [model.state() for _ in range(n_steps)]

    # Simulate the model
    for i in tqdm(range(n_steps - 1), desc="Simulating"):
        collide(model, states[i], collision_margin=0.0)
        engine.simulate(states[i], states[i + 1], control, dt)

    print(f"Saving simulation to {output_file}")
    save_simulation(model, states, output_file)


if __name__ == "__main__":
    main()
