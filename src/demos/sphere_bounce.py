import os

import matplotlib.pyplot as plt
import torch
from demos.utils import save_simulation
from pbd_torch.collision import collide
from pbd_torch.constants import ROT_IDENTITY
from pbd_torch.model import Model
from pbd_torch.model import Quaternion
from pbd_torch.model import Vector3
from pbd_torch.newton_engine import NonSmoothNewtonEngine
from pbd_torch.terrain import create_terrain_from_exr_file
from pbd_torch.xpbd_engine import XPBDEngine
from tqdm import tqdm

ENGINE = "nsn"  # Choose from ["nsn", "xpbd"]
EXPERIMENT_NAME = f"sphere_bounce_{ENGINE}"

# Initial state
INITIAL_POSITION = torch.tensor([-0.5, -0.0, 4.5]).view(3, 1)
INITIAL_VELOCITY = torch.tensor([5.0, 0.0, -5.0]).view(3, 1)

# Simulation Constants
MAX_CONTACTS_PER_BODY = 36
N_COLLISION_POINTS = 2000
RESTITUTION = 1.0
DYNAMIC_FRICTION = 1.0

# Get the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "..", "data")
experiment_dir = os.path.abspath(os.path.join(data_dir, EXPERIMENT_NAME))

# Output File Names
TIME_FILE = os.path.join(experiment_dir, "time.pt")
TRAJECTORY_FILE = os.path.join(experiment_dir, "trajectory.pt")

USE_TERRAIN = False
TERRAIN_FILE = os.path.join(data_dir, "blender", "ground4", "textures", "ant01.002_heightmap.exr")
PLOT_FILE = os.path.join(experiment_dir, "gradients_and_loss.png")

def create_model(device: torch.device) -> tuple[Model, int]:
    terrain = create_terrain_from_exr_file(
        heightmap_path=TERRAIN_FILE,
        size_x=40.0,
        size_y=40.0,
        device=device,
    )
    if USE_TERRAIN:
        model = Model(terrain=terrain, max_contacts_per_body=MAX_CONTACTS_PER_BODY)
    else:
        model = Model(max_contacts_per_body=MAX_CONTACTS_PER_BODY)

    sphere_idx = model.add_sphere(
        m=1.0,
        radius=1.0,
        name="sphere",
        pos=Vector3(INITIAL_POSITION.view(-1)),
        rot=Quaternion(ROT_IDENTITY),
        restitution=RESTITUTION,
        dynamic_friction=DYNAMIC_FRICTION,
        n_collision_points=N_COLLISION_POINTS,
    )
    return model, sphere_idx


def simulate_sphere(
    model: Model,
    sphere_idx: int,
    dt: float,
    n_steps: int,
    device: torch.device,
) -> torch.Tensor:
    model.body_q[sphere_idx, :3] = INITIAL_POSITION
    model.body_qd[sphere_idx, 3:] = INITIAL_VELOCITY

    if ENGINE == "xpbd":
        engine = XPBDEngine(model, pos_iters=5)
    elif ENGINE == "nsn":
        engine = NonSmoothNewtonEngine(model, iterations=150)
    else:
        raise ValueError(f"Unknown engine: {ENGINE}")

    control = model.control()
    states = [model.state() for _ in range(n_steps)]

    t = []
    q = []

    for i in tqdm(range(n_steps - 1), desc="Simulating"):
        collide(model, states[i], collision_margin=0.0)

        if ENGINE == "xpbd":
            engine.simulate(states[i], states[i + 1], control, dt)
        elif ENGINE == "nsn":
            engine.simulate(states[i], states[i + 1], control, dt)
        else:
            raise ValueError(f"Unknown engine: {ENGINE}")

        t.append(i * dt)
        q.append(states[i].body_q[sphere_idx, :3].view(-1).cpu().numpy())

    # Save time and trajectory data
    torch.save(torch.tensor(t), TIME_FILE)
    torch.save(torch.tensor(q), TRAJECTORY_FILE)


def visualize_trajectory():
    # Load time and trajectory data
    time = torch.load(TIME_FILE).cpu().numpy()
    trajectory = torch.load(TRAJECTORY_FILE).cpu().numpy()

    # Unpack x, y, z coordinates
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

    # Plot trajectory
    plt.figure(figsize=(10, 6))
    plt.plot(time, x, label="x-coordinate", color="r")
    plt.plot(time, y, label="y-coordinate", color="g")
    plt.plot(time, z, label="z-coordinate", color="b")
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.title("Sphere Trajectory Over Time")
    plt.legend()
    plt.grid()
    plt.savefig(PLOT_FILE)  # Save the plot as an image file
    plt.show()


def main():
    dt = 0.001
    n_steps = 1000
    device = torch.device("cpu")

    # Initialize model and sphere
    model, sphere_idx = create_model(device)

    os.makedirs(experiment_dir, exist_ok=True)

    # Run simulation
    target_transform = simulate_sphere(model, sphere_idx, dt, n_steps, device)

    # Visualize results
    visualize_trajectory()


if __name__ == "__main__":
    main()
