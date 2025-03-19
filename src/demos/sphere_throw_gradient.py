from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from demos.utils import save_simulation
from pbd_torch.collision import collide
from pbd_torch.constants import ROT_IDENTITY
from pbd_torch.model import Model
from pbd_torch.model import Quaternion
from pbd_torch.model import State
from pbd_torch.model import Vector3
from pbd_torch.xpbd_engine import XPBDEngine
from tqdm import tqdm

matplotlib.use("TkAgg")

# Detect anomalies in torch autograd
torch.autograd.set_detect_anomaly(True)


def trajectory_loss(states: List[State], target_states: List[State]):
    loss = torch.tensor([0.0], device=states[0].body_q.device, requires_grad=True)
    for state, target_state in zip(states, target_states):
        loss = loss + torch.sum((state.body_q[0][:3] - target_state.body_q[0][:3]) ** 2)
    loss = loss / len(states)
    return loss


def plot_trajectory(states: List[State], target_states: List[State] = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    trajectory = torch.stack([state.body_q[0][:3].cpu().detach() for state in states])
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])

    if target_states is not None:
        target_trajectory = torch.stack(
            [state.body_q[0][:3].cpu().detach() for state in target_states]
        )
        ax.plot(
            target_trajectory[:, 0], target_trajectory[:, 1], target_trajectory[:, 2]
        )

    # Make all axes equal
    max_range = torch.max(trajectory.max(0)[0] - trajectory.min(0)[0])
    mid_x = (trajectory.max(0)[0][0] + trajectory.min(0)[0][0]) / 2
    mid_y = (trajectory.max(0)[0][1] + trajectory.min(0)[0][1]) / 2
    mid_z = (trajectory.max(0)[0][2] + trajectory.min(0)[0][2]) / 2
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    plt.show()


class BallThrow:

    def __init__(self, dt: float, n_steps: int, output_file: str):
        self.dt = dt
        self.n_steps = n_steps
        self.output_file = output_file

        self.model = Model(
            device=(
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            ),
            requires_grad=True,
        )
        self.integrator = XPBDEngine(iterations=2)

        self.sphere = self.model.add_sphere(
            m=10.0,
            radius=1.0,
            name="sphere",
            pos=Vector3(torch.tensor([0.0, 0.0, 3.5])),
            rot=Quaternion(ROT_IDENTITY),
            restitution=1.0,
            dynamic_friction=1.0,
            n_collision_points=400,
        )

        # Set up the initial state
        self.states = [self.model.state() for _ in range(n_steps)]
        self.target_states = [self.model.state() for _ in range(n_steps)]

        self.control = self.model.control()

    def _reset(self):
        self.loss = 0.0
        self.states = [self.model.state() for _ in range(self.n_steps)]

    def forward(self, states: List[State]):
        for i in tqdm(range(len(states) - 1), desc="Simulating"):
            collide(self.model, states[i], collision_margin=0.1)
            self.integrator.simulate(
                self.model, states[i], states[i + 1], self.control, self.dt
            )

    def generate_target_states(self):
        self.target_states[0].body_qd[0][3:] = torch.tensor(
            [2.5, 0.0, 0.0], device=self.model.device
        )
        for i in tqdm(
            range(len(self.target_states) - 1), desc="Generating target states"
        ):
            collide(self.model, self.target_states[i], collision_margin=0.1)
            self.integrator.simulate(
                self.model,
                self.target_states[i],
                self.target_states[i + 1],
                self.control,
                self.dt,
            )

        save_simulation(self.model, self.target_states, self.output_file)

    def run(self):
        vx_samples = torch.linspace(0, 5, 60)
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
            grads.append(self.states[0].body_qd.grad[0][3].cpu().detach().item())
            losses.append(loss.cpu().detach().item())
            print(grads)

        # Save the grads and losses to the .npz file
        x = vx_samples.cpu().detach().numpy()
        grads = np.array(grads)
        losses = np.array(losses)
        np.savez("sphere_throw_data_2.npz", x=x, grads=grads, losses=losses)


def plot_data(file: str):
    data = np.load(file)
    x = data["x"]
    grads = data["grads"]
    losses = data["losses"]

    # Two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Velocity vs Loss and Gradient")

    ax1.plot(x, losses, label="Loss")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Velocity")
    ax1.legend()

    ax2.plot(x, grads, label="Gradient", color="orange")
    ax2.set_ylabel("Gradient")
    ax2.set_xlabel("Velocity")
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

    # def plot_loss(self):
    #     forces = np.linspace(0, 200, 50)
    #     losses = np.zeros_like(forces)
    #     grads = np.zeros_like(forces)
    #
    #     for i, f_x in enumerate(forces):
    #         print(f"Iteration {i}")
    #         force = wp.array([[0.0, 0.0, 0.0, f_x, 0.0, 0.0]], dtype=wp.spatial_vectorf, requires_grad=True)
    #         losses[i], grads[i] = self.step(force)
    #
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    #
    #     # Plot the loss curve
    #     ax1.plot(forces, losses, label="Loss")
    #     ax1.set_xlabel("Force")
    #     ax1.set_ylabel("Loss")
    #     ax1.set_title("Loss vs Force")
    #     ax1.legend()
    #
    #     # Make sure that that grads are not too large
    #     grads = np.clip(grads, -1e4, 1e4)
    #
    #     # Plot the gradient curve
    #     ax2.plot(forces, grads, label="Gradient", color="orange")
    #     ax2.set_xlabel("Force")
    #     ax2.set_ylabel("Gradient")
    #     ax2.set_title("Gradient vs Force")
    #     ax2.legend()
    #
    #     plt.suptitle("Loss and Gradient vs Force")
    #     plt.tight_layout(rect=[0, 0, 1, 0.95])
    #     plt.show()


if __name__ == "__main__":
    model = BallThrow(0.025, 150, "simulation/sphere_throw_gradient.json")
    model.generate_target_states()
    plot_trajectory(model.target_states)
    model.run()

    # plot_data("sphere_throw_data_2.npz")
