from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from demos.utils import save_simulation
from pbd_torch.collision import collide
from pbd_torch.constants import ROT_90_X
from pbd_torch.constants import ROT_IDENTITY
from pbd_torch.constants import ROT_NEG_90_X
from pbd_torch.integrator import XPBDIntegrator
from pbd_torch.model import Model
from pbd_torch.model import Quaternion
from pbd_torch.model import State
from pbd_torch.model import Vector3
from tqdm import tqdm

matplotlib.use('TkAgg')


def trajectory_loss(states: List[State],
                    target_states: List[State],
                    body_idx: int = 0):
    loss = torch.tensor([0.0],
                        device=states[0].body_q.device,
                        requires_grad=True)
    for state, target_state in zip(states, target_states):
        loss = loss + torch.sum(
            (state.body_q[body_idx][:3] - target_state.body_q[body_idx][:3])**
            2)
    loss = loss / len(states)
    return loss


def plot_trajectory(states: List[State],
                    target_states: List[State] = None,
                    body_idx: int = 0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    trajectory = torch.stack(
        [state.body_q[body_idx][:3].cpu().detach() for state in states])
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])

    if target_states is not None:
        target_trajectory = torch.stack([
            state.body_q[body_idx][:3].cpu().detach()
            for state in target_states
        ])
        ax.plot(target_trajectory[:, 0], target_trajectory[:, 1],
                target_trajectory[:, 2])

    # Make all axes equal
    max_range = torch.max(trajectory.max(0)[0] - trajectory.min(0)[0])
    mid_x = (trajectory.max(0)[0][0] + trajectory.min(0)[0][0]) / 2
    mid_y = (trajectory.max(0)[0][1] + trajectory.min(0)[0][1]) / 2
    mid_z = (trajectory.max(0)[0][2] + trajectory.min(0)[0][2]) / 2
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    plt.show()


class HelhestDrive:

    def __init__(self, dt: float, n_steps: int, output_file: str):
        self.dt = dt
        self.n_steps = n_steps
        self.output_file = output_file

        self.model = Model(device=torch.device('cuda') if
                           torch.cuda.is_available() else torch.device('cpu'),
                           requires_grad=True)
        self.integrator = XPBDIntegrator(iterations=2)

        # Add robot base
        self.base = self.model.add_box(m=10.0,
                                       hx=1.0,
                                       hy=2.0,
                                       hz=1.0,
                                       name='box',
                                       pos=Vector3(
                                           torch.tensor([0.0, 0.0, 2.1])),
                                       rot=Quaternion(ROT_IDENTITY),
                                       n_collision_points=200,
                                       restitution=0.1,
                                       dynamic_friction=1.0)

        # Add left wheel
        self.left_wheel = self.model.add_cylinder(
            m=1.0,
            radius=2.0,
            height=0.4,
            name='left_wheel',
            pos=Vector3(torch.tensor([0.0, 3.0, 2.1])),
            rot=Quaternion(ROT_NEG_90_X),
            n_collision_points_base=64,
            n_collision_points_surface=64,
            restitution=0.1,
            dynamic_friction=1.0)

        # Add right wheel
        self.right_wheel = self.model.add_cylinder(
            m=1.0,
            radius=2.0,
            height=0.4,
            name='right_wheel',
            pos=Vector3(torch.tensor([0.0, -3.0, 2.1])),
            rot=Quaternion(ROT_90_X),
            n_collision_points_base=64,
            n_collision_points_surface=64,
            restitution=0.1,
            dynamic_friction=1.0)

        # Back wheel
        self.back_wheel = self.model.add_cylinder(
            m=1.0,
            radius=2.0,
            height=0.4,
            name='back_wheel',
            pos=Vector3(torch.tensor([-4.0, 0.0, 2.1])),
            rot=Quaternion(ROT_NEG_90_X),
            n_collision_points_base=64,
            n_collision_points_surface=64)

        # Add left hinge joint
        self.left_wheel_joint = self.model.add_hinge_joint(
            parent=self.base,
            child=self.left_wheel,
            axis=Vector3(torch.tensor([0.0, 1.0, 0.0])),
            name='left_wheel_joint',
            parent_xform=torch.tensor([0.0, 2.5, 0.0, 1.0, 0.0, 0.0, 0.0]),
            child_xform=torch.cat((torch.tensor([0.0, 0.0, -0.5]), ROT_90_X)))

        # Add right hinge joint
        self.right_wheel_joint = self.model.add_hinge_joint(
            parent=self.base,
            child=self.right_wheel,
            axis=Vector3(torch.tensor([0.0, 1.0, 0.0])),
            name='right_wheel_joint',
            parent_xform=torch.tensor([0.0, -2.5, 0.0, 1.0, 0.0, 0.0, 0.0]),
            child_xform=torch.cat((torch.tensor([0.0, 0.0,
                                                 -0.5]), ROT_NEG_90_X)))

        # Add back hinge joint
        self.back_wheel_joint = self.model.add_hinge_joint(
            parent=self.base,
            child=self.back_wheel,
            axis=Vector3(torch.tensor([0.0, 1.0, 0.0])),
            name='back_wheel_joint',
            parent_xform=torch.tensor([-4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            child_xform=torch.cat((torch.tensor([0.0, 0.0, 0.0]), ROT_90_X)))

        # Set up the initial state
        self.states = [self.model.state() for _ in range(n_steps)]
        self.target_states = [self.model.state() for _ in range(n_steps)]

        self.control = self.model.control()

    def _reset(self):
        self.loss = 0.0
        self.states = [self.model.state() for _ in range(self.n_steps)]

    def forward(self, states: List[State]):
        for i in tqdm(range(len(states) - 1), desc='Simulating'):
            collide(self.model, states[i], collision_margin=0.1)
            self.integrator.simulate(self.model, states[i], states[i + 1],
                                     self.control, self.dt)

    def generate_target_states(self):
        self.target_states[0].body_qd[0][3:] = torch.tensor(
            [2.5, 0.0, 0.0], device=self.model.device)
        for i in tqdm(range(len(self.target_states) - 1),
                      desc='Generating target states'):
            collide(self.model, self.target_states[i], collision_margin=0.1)
            self.integrator.simulate(self.model, self.target_states[i],
                                     self.target_states[i + 1], self.control,
                                     self.dt)

        save_simulation(self.model, self.target_states, self.output_file)

    def run(self):
        vx_samples = torch.linspace(0, 5, 12)
        grads = []
        losses = []
        for i in range(len(vx_samples)):
            self._reset()
            self.states[0].body_qd[0][3] = vx_samples[i]
            self.states[0].body_qd.retain_grad()
            self.forward(self.states)
            # plot_trajectory(self.states, self.target_states)
            loss = trajectory_loss(self.states, self.target_states)
            print(f"Vx: {vx_samples[i]}, Loss: {loss}")
            loss.backward(retain_graph=True)

            # Update the velocity
            grads.append(
                self.states[0].body_qd.grad[0][3].cpu().detach().item())
            losses.append(loss.cpu().detach().item())
            print(grads)

        # Save the grads and losses to the .npz file
        x = vx_samples.cpu().detach().numpy()
        grads = np.array(grads)
        losses = np.array(losses)
        np.savez('helhest_gradient_data.npz', x=x, grads=grads, losses=losses)


def plot_data(file: str):
    data = np.load(file)
    x = data['x']
    grads = data['grads']
    losses = data['losses']

    # Two subplots
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Velocity vs Loss and Gradient')

    ax1.plot(x, losses)
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Vx')

    ax2.plot(x, grads)
    ax2.set_ylabel('Gradient')
    ax2.set_xlabel('Vx')

    plt.show()


if __name__ == '__main__':
    model = HelhestDrive(0.01, 50, 'simulation/helhest_gradient.json')
    # model.generate_target_states()
    # # plot_trajectory(model.target_states)
    # model.run()

    plot_data('helhest_gradient_data.npz')
