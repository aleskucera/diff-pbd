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
    output_file = os.path.join('simulation', 'cylinder.json')

    model = Model()
    integrator = XPBDIntegrator(iterations=2)

    # Add robot base
    box = model.add_cylinder(m=1.0,
                             radius=1.0,
                             height=0.2,
                             name='cylinder',
                             pos=Vector3(torch.tensor([0.0, 0.0, 3.5])),
                             rot=Quaternion(ROT_IDENTITY),
                             n_collision_points_base=64,
                             n_collision_points_surface=128)

    # Add initial rotation to the box
    model.body_qd[box, :3] = torch.tensor([0.5, 1.0, 0.0])
    model.body_qd[box, 3:] = torch.tensor([0.0, 0.0, 0.0])

    # Set up the initial state
    states = [model.state() for _ in range(n_steps)]

    control = model.control()

    states[0].time = time

    # Simulate the model
    for i in tqdm(range(n_steps - 1), desc='Simulating'):
        time += dt
        states[i + 1].time = time
        integrator.simulate(model, states[i], states[i + 1], control, dt)

    print(f'Saving simulation to {output_file}')
    save_simulation(model, states, output_file)


if __name__ == '__main__':
    main()
