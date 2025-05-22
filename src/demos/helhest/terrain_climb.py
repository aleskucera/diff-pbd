import os

import torch
from demos.utils import save_simulation
from pbd_torch.collision import collide, collide_terrain_friction
from pbd_torch.newton_engine_friction import NonSmoothNewtonEngine
from pbd_torch.xpbd_engine import XPBDEngine
from pbd_torch.terrain import create_terrain_from_exr_file
from demos.helhest.utils import create_helhest_model
from tqdm import tqdm

ENGINE = "XPBD"  # Choose between "XPBD" and "NonSmoothNewton"

# Get the file path of the current script
curr_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(curr_dir, "..", "..", ".."))
OUTPUT_FILE = os.path.join(project_dir, "simulation", "helhest_terrain_climb.json")
HEIGHT_MAP = os.path.join(project_dir, "data", "blender", "ground4", "textures", "ant01.002_heightmap.exr")


def main():
    dt = 0.01
    n_steps = 200
    device = torch.device("cpu")
    collision_margin = 0.0
    friction_collision_margin = 0.08

    terrain = create_terrain_from_exr_file(
        heightmap_path=HEIGHT_MAP,
        size_x=40.0,
        size_y=40.0,
        device=device,
    )

    model, idxs = create_helhest_model(
        base_pos=(-8.0, 0.0, 3.0),
        device=device,
        terrain=terrain,
        max_contacts_per_body=16,
    )

    if ENGINE == "XPBD":
        engine = XPBDEngine(model, pos_iters=5, device=device)
    elif ENGINE == "NonSmoothNewton":
        engine = NonSmoothNewtonEngine(model, iterations=100, device=device)
    else:
        raise ValueError(f"Unknown engine: {ENGINE}")

    control = model.control()
    states = [model.state() for _ in range(n_steps)]

    for i in tqdm(range(n_steps - 1), desc="Simulating"):
        collide(model, states[i], collision_margin=collision_margin)
        collide_terrain_friction(model, states[i], collision_margin=friction_collision_margin)

        control.add_actuation(idxs['left_wheel_joint'], 1.5)
        control.add_actuation(idxs['right_wheel_joint'], -1.5)

        engine.simulate(states[i], states[i + 1], control, dt)

    print(f"Saving simulation to {OUTPUT_FILE}")
    save_simulation(model, states, OUTPUT_FILE)


if __name__ == "__main__":
    main()
