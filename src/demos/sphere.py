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
    n_steps = 2000
    output_file = os.path.join('simulation', 'sphere.json')

    model = Model()
    integrator = XPBDIntegrator(iterations=2)

    sphere = model.add_sphere(m=10.0,
                              radius=1.0,
                              name='sphere',
                              pos=Vector3(torch.tensor([0.0, 0.0, 3.5])),
                              rot=Quaternion(ROT_IDENTITY),
                              restitution=0.8,
                              dynamic_friction=1.0,
                              n_collision_points=400)

    # Add initial rotation to the box
    model.body_qd[sphere, :3] = torch.tensor([0.0, 0.0, 0.0])
    model.body_qd[sphere, 3:] = torch.tensor([1.0, 0.0, 2.0])

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
