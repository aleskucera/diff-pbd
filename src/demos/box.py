import os

import torch
from pbd_torch.constants import *
from pbd_torch.integrator import XPBDIntegrator
from pbd_torch.model import *
from pbd_torch.transform import *
from pbd_torch.utils import *
from tqdm import tqdm


def main():
    time = 0.0
    dt = 0.01
    n_steps = 200
    output_file = os.path.join('simulation', 'box.json')

    model = Model()
    integrator = XPBDIntegrator(iterations=1)

    # Add robot base
    box = model.add_box(m=1.0,
                        hx=1.0,
                        hy=1.0,
                        hz=1.0,
                        name='box',
                        pos=Vector3(torch.tensor([0.0, 0.0, 3.5])),
                        rot=Quaternion(ROT_IDENTITY),
                        n_collision_points=200)

    # Add initial rotation to the box
    model.body_qd[box, :3] = torch.tensor([1.0, 2.0, 0.0])
    model.body_qd[box, 3:] = torch.tensor([0.0, 0.0, 0.0])

    # Set up the initial state
    states = [model.state() for _ in range(n_steps)]

    control = model.control()

    states[0].time = time

    # Simulate the model
    for i in tqdm(range(n_steps - 1), desc='Simulating'):
        states[i + 1].time = time
        integrator.simulate(model, states[i], states[i + 1], control, dt)
        time += dt

    print(f'Saving simulation to {output_file}')
    save_simulation(model, states, output_file)


if __name__ == '__main__':
    main()
