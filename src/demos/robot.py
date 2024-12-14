import os

import torch
from pbd_torch.animation import AnimationController2
from pbd_torch.constants import *
from pbd_torch.model import *
from pbd_torch.robot_integrator import RobotIntegrator
from pbd_torch.transform import *
from pbd_torch.utils import *
from tqdm import tqdm


def main():
    time = 0.0
    dt = 0.01
    n_steps = 100
    output_file = os.path.join('simulation', 'robot.json')

    model = Model()
    integrator = RobotIntegrator()

    # Add robot base
    base = model.add_box(m=1.0,
                         hx=1.0,
                         hy=1.0,
                         hz=1.0,
                         name='box',
                         pos=Vector3(torch.tensor([0.0, 0.0, 2.5])),
                         rot=Quaternion(ROT_IDENTITY),
                         n_collision_points=200)

    # Add left wheel
    left_wheel = model.add_cylinder(m=1.0,
                                    radius=2.0,
                                    height=0.4,
                                    name='left_wheel',
                                    pos=Vector3(torch.tensor([0.0, 1.6, 2.5])),
                                    rot=Quaternion(ROT_NEG_90_X),
                                    n_collision_points_base=64,
                                    n_collision_points_surface=64)

    # Add right wheel
    right_wheel = model.add_cylinder(m=1.0,
                                     radius=2.0,
                                     height=0.4,
                                     name='right_wheel',
                                     pos=Vector3(torch.tensor([0.0, -1.6,
                                                               2.5])),
                                     rot=Quaternion(ROT_90_X),
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
                          pos=Vector3(torch.Tensor([0.0, 0.0, 2.5])),
                          rot=Quaternion(ROT_IDENTITY),
                          body_indices=[base, left_wheel, right_wheel],
                          joint_indices=[left_wheel_joint, right_wheel_joint])

    # Set up the initial state
    states = [model.state() for _ in range(n_steps)]

    control = model.control()

    states[0].time = time
    for i in tqdm(range(n_steps - 1), desc='Simulating'):
        time += dt
        states[i + 1].time = time
        integrator.simulate(model, states[i], states[i + 1], control, dt)

    print(f'Saving simulation to {output_file}')
    save_simulation(model, states, output_file)

    # animator = AnimationController2(model, states)
    # animator.start()


if __name__ == '__main__':
    main()
