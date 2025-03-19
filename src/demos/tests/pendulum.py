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
    dt = 0.001
    n_steps = 5000
    output_file = os.path.join('simulation', 'pendulum.json')

    model = Model()
    integrator = XPBDIntegrator(iterations=2)

    upper_sphere = model.add_sphere(m=1.0,
                                    radius=0.3,
                                    name='upper_sphere',
                                    pos=Vector3(torch.tensor([0.0, 0.0, 3.0])),
                                    rot=Quaternion(ROT_IDENTITY),
                                    restitution=1.0,
                                    dynamic_friction=1.0,
                                    n_collision_points=200)

    upper_joint = model.add_hinge_joint(
        parent=-1,
        child=upper_sphere,
        axis=Vector3(torch.tensor([0.0, 1.0, 0.0])),
        name='upper_joint',
        parent_xform=torch.tensor([0.0, 0.0, 5.0, 1.0, 0.0, 0.0, 0.0]),
        child_xform=torch.cat((torch.tensor([0.0, 0.0, 2.0]), ROT_IDENTITY)))

    lower_sphere = model.add_sphere(m=1.0,
                                    radius=0.3,
                                    name='lower_sphere',
                                    pos=Vector3(torch.tensor([0.0, 0.0, 1.0])),
                                    rot=Quaternion(ROT_IDENTITY),
                                    restitution=1.0,
                                    dynamic_friction=1.0,
                                    n_collision_points=200)

    lower_joint = model.add_hinge_joint(
        parent=upper_sphere,
        child=lower_sphere,
        axis=Vector3(torch.tensor([0.0, 1.0, 0.0])),
        name='lower_joint',
        parent_xform=torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        child_xform=torch.cat((torch.tensor([0.0, 0.0, 2.0]), ROT_IDENTITY)))

    model.body_qd[upper_sphere, :3] = torch.tensor([0.0, 0.0, 0.0])
    model.body_qd[upper_sphere, 3:] = torch.tensor([3.0, 0.0, 0.0])

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
