import os

import torch
from demos.utils import save_simulation
from pbd_torch.collision import collide
from pbd_torch.constants import ROT_IDENTITY
from pbd_torch.integrator import XPBDIntegrator
from pbd_torch.model import Model
from pbd_torch.model import Quaternion
from pbd_torch.model import Vector3
from tqdm import tqdm


def main():
    dt = 0.01
    n_steps = 200
    output_file = os.path.join('simulation', 'box.json')

    model = Model()
    integrator = XPBDIntegrator(iterations=4)

    # Add robot base
    box = model.add_box(m=1.0,
                        hx=0.5,
                        hy=0.5,
                        hz=0.5,
                        name='box',
                        pos=Vector3(torch.tensor([0.0, 0.0, 3.5])),
                        rot=Quaternion(ROT_IDENTITY),
                        restitution=1.0,
                        dynamic_friction=1.0,
                        n_collision_points=200)

    # Add initial rotation to the box
    model.body_qd[box, :3] = torch.tensor([1.0, 2.0, 0.0])
    model.body_qd[box, 3:] = torch.tensor([0.0, 0.0, 0.0])

    # Set up the initial state
    states = [model.state() for _ in range(n_steps)]

    control = model.control()

    # Simulate the model
    for i in tqdm(range(n_steps - 1), desc='Simulating'):
        collide(model, states[i], collision_margin=0.1)
        integrator.simulate(model, states[i], states[i + 1], control, dt)

    print(f'Saving simulation to {output_file}')
    save_simulation(model, states, output_file)


if __name__ == '__main__':
    main()
