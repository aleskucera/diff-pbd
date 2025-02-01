import os

import torch
from demos.utils import save_simulation
from pbd_torch.collision import collide
from pbd_torch.constants import ROT_90_X
from pbd_torch.constants import ROT_IDENTITY
from pbd_torch.constants import ROT_NEG_90_X
from pbd_torch.model import Model
from pbd_torch.model import Quaternion
from pbd_torch.model import Vector3
from pbd_torch.robot_integrator import RobotIntegrator
from tqdm import tqdm


def main():
    dt = 0.01
    n_steps = 500
    output_file = os.path.join('simulation', 'robot.json')

    model = Model()
    integrator = RobotIntegrator(iterations=2)

    # Add robot base
    base = model.add_box(m=10.0,
                         hx=1.0,
                         hy=2.0,
                         hz=1.0,
                         name='box',
                         pos=Vector3(torch.tensor([0.0, 0.0, 2.1])),
                         rot=Quaternion(ROT_IDENTITY),
                         n_collision_points=200)

    # Add left wheel
    left_wheel = model.add_cylinder(m=1.0,
                                    radius=2.0,
                                    height=0.4,
                                    name='left_wheel',
                                    pos=Vector3(torch.tensor([0.0, 2.6, 2.1])),
                                    rot=Quaternion(ROT_NEG_90_X),
                                    n_collision_points_base=64,
                                    n_collision_points_surface=64)

    # Add right wheel
    right_wheel = model.add_cylinder(m=1.0,
                                     radius=2.0,
                                     height=0.4,
                                     name='right_wheel',
                                     pos=Vector3(torch.tensor([0.0, -2.6,
                                                               2.1])),
                                     rot=Quaternion(ROT_90_X),
                                     n_collision_points_base=64,
                                     n_collision_points_surface=64)

    # Back wheel
    back_wheel = model.add_cylinder(m=1.0,
                                    radius=2.0,
                                    height=0.4,
                                    name='back_wheel',
                                    pos=Vector3(torch.tensor([-4.0, 0.0,
                                                              2.1])),
                                    rot=Quaternion(ROT_NEG_90_X),
                                    n_collision_points_base=64,
                                    n_collision_points_surface=64)

    # Add left hinge joint
    left_wheel_joint = model.add_hinge_joint(parent=base,
                                             child=left_wheel,
                                             axis=Vector3(
                                                 torch.tensor([0.0, -1.0,
                                                               0.0])),
                                             name='left_wheel_joint')

    # Add right hinge joint
    right_wheel_joint = model.add_hinge_joint(parent=base,
                                              child=right_wheel,
                                              axis=Vector3(
                                                  torch.tensor([0.0, 1.0,
                                                                0.0])),
                                              name='right_wheel_joint')

    # Create robot
    model.construct_robot(name='robot',
                          pos=Vector3(torch.Tensor([-2.0, 0.0, 2.1])),
                          rot=Quaternion(ROT_IDENTITY),
                          body_indices=[base, left_wheel, right_wheel],
                          joint_indices=[left_wheel_joint, right_wheel_joint])

    # Set up the initial state
    states = [model.state() for _ in range(n_steps)]

    control = model.control()

    for i in tqdm(range(n_steps - 1), desc='Simulating'):
        collide(model, states[i], collision_margin=0.1)
        states[i].body_qd[1, :3] = torch.tensor([0.0, 0.0, 10.0])
        states[i].body_qd[2, :3] = torch.tensor([0.0, 0.0, -10.0])
        integrator.simulate(model, states[i], states[i + 1], control, dt)

    print(f'Saving simulation to {output_file}')
    save_simulation(model, states, output_file)


if __name__ == '__main__':
    main()
