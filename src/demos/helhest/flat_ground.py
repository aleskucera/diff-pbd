import os

import torch
from tqdm import tqdm
from demos.utils import save_simulation
from pbd_torch.collision import collide
from pbd_torch.xpbd_engine import XPBDEngine
from pbd_torch.newton_engine import NonSmoothNewtonEngine
from demos.helhest.utils import create_helhest_model

ENGINE = "NonSmoothNewton"  # Choose between "XPBD" and "NonSmoothNewton"

def main():
    dt = 0.001
    n_steps = 500
    device = torch.device("cuda")
    collision_margin = 0.0
    output_file = os.path.join("simulation", "helhest_flat_ground.json")

    model, idxs = create_helhest_model(
        base_pos=(-8.0, 0.0, 3.0),
        device=device,
        terrain=None,
        max_contacts_per_body=16,
    )

    if ENGINE == "XPBD":
        engine = XPBDEngine(model, pos_iters=50, device=device)
    elif ENGINE == "NonSmoothNewton":
        engine = NonSmoothNewtonEngine(model, iterations=50, device=device)
    else:
        raise ValueError(f"Unknown engine: {ENGINE}")

    states = [model.state() for _ in range(n_steps)]
    control = model.control()

    for i in tqdm(range(n_steps - 1), desc="Simulating"):
        collide(model, states[i], collision_margin=collision_margin)

        control.add_actuation(idxs['left_wheel_joint'], 3.0)
        control.add_actuation(idxs['right_wheel_joint'], -3.0)

        engine.simulate(states[i], states[i + 1], control, dt)

        # print(f"{states[i].joint_qd}")

    print(f"Saving simulation to {output_file}")
    save_simulation(model, states, output_file)


if __name__ == "__main__":
    main()
