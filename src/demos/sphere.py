import os

import torch
from demos.utils import save_simulation
from pbd_torch.collision import collide
from pbd_torch.collision import collide_batch
from pbd_torch.constants import ROT_IDENTITY
from pbd_torch.model import Model
from pbd_torch.model import Quaternion
from pbd_torch.model import Vector3
from pbd_torch.newton_engine import NonSmoothNewtonEngine
from pbd_torch.xpbd_engine import XPBDEngine
from tqdm import tqdm


def main():
    dt = 0.005
    n_steps = 800
    output_file = os.path.join("simulation", "sphere.json")

    model = Model()
    engine = XPBDEngine(iterations=2)
    engine2 = NonSmoothNewtonEngine(iterations=10)

    sphere = model.add_sphere(
        m=10.0,
        radius=1.0,
        name="sphere",
        pos=Vector3(torch.tensor([0.0, 0.0, 3.0])),
        rot=Quaternion(ROT_IDENTITY),
        restitution=1.0,
        dynamic_friction=1.0,
        n_collision_points=400,
    )

    # Add initial rotation to the box
    model.body_qd[sphere, :3] = torch.tensor([0.0, 0.0, 0.0])
    model.body_qd[sphere, 3:] = torch.tensor([0.0, 0.0, 0.0])

    # Set up the initial state
    states = [model.state() for _ in range(n_steps)]

    control = model.control()

    # Simulate the model
    for i in tqdm(range(n_steps - 1), desc="Simulating"):
        collide(model, states[i], collision_margin=0.1)
        collide_batch(model, states[i], collision_margin=0.1)
        # engine.simulate(model, states[i], states[i + 1], control, dt)

        engine2.simulate(model, states[i], states[i + 1], control, dt)

    print(f"Saving simulation to {output_file}")
    save_simulation(model, states, output_file)


if __name__ == "__main__":
    main()
