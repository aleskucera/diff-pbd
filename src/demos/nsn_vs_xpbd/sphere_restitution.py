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

ENGINES = ["nsn", "xpbd"]
EXPERIMENT_NAME = "sphere_restitution_comparison"

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
data_dir = os.path.join(current_dir, "..", "..", "..", "data")
experiment_dir = os.path.abspath(os.path.join(data_dir, EXPERIMENT_NAME))

# Output File Names
TIME_FILE = os.path.join(experiment_dir, "time.pt")
TRAJECTORY_FILES = {
    engine: os.path.join(experiment_dir, f"trajectory_{engine}.pt") for engine in ENGINES
}
PLOT_FILE = os.path.join(experiment_dir, "trajectories_comparison.png")

USE_TERRAIN = False
TERRAIN_FILE = os.path.join(data_dir, "blender", "ground4", "textures", "ant01.002_heightmap.exr")


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
        engine_type: str
) -> torch.Tensor:
    model.body_q[sphere_idx, :3] = INITIAL_POSITION
    model.body_qd[sphere_idx, 3:] = INITIAL_VELOCITY

    if engine_type == "xpbd":
        engine = XPBDEngine(model, pos_iters=5)
    elif engine_type == "nsn":
        engine = NonSmoothNewtonEngine(model, iterations=150)
    else:
        raise ValueError(f"Unknown engine: {engine_type}")

    control = model.control()
    states = [model.state() for _ in range(n_steps)]

    t = []
    q = []

    for i in tqdm(range(n_steps - 1), desc=f"Simulating {engine_type}"):
        collide(model, states[i], collision_margin=0.0)

        engine.simulate(states[i], states[i + 1], control, dt)

        t.append(i * dt)
        q.append(states[i].body_q[sphere_idx, :3].view(-1).cpu().numpy())

    # Save time and trajectory data
    torch.save(torch.tensor(t), TIME_FILE)
    torch.save(torch.tensor(q), TRAJECTORY_FILES[engine_type])


def visualize_trajectories():
    # Load time and trajectory data
    time = torch.load(TIME_FILE).cpu().numpy()

    # Create subplots for x, y, z coordinates
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    colors = {'nsn': 'blue', 'xpbd': 'red'}
    linestyles = {'nsn': '-', 'xpbd': '--'}

    for engine in ENGINES:
        trajectory = torch.load(TRAJECTORY_FILES[engine]).cpu().numpy()
        x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

        # Plot x-coordinate
        ax1.plot(time, x, label=f'x ({engine.upper()})',
                 color=colors[engine], linestyle=linestyles[engine])

        # Plot y-coordinate
        ax2.plot(time, y, label=f'y ({engine.upper()})',
                 color=colors[engine], linestyle=linestyles[engine])

        # Plot z-coordinate
        ax3.plot(time, z, label=f'z ({engine.upper()})',
                 color=colors[engine], linestyle=linestyles[engine])

    # Configure plots
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('X Position')
    ax1.set_title('X-Coordinate Comparison')
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Y Position')
    ax2.set_title('Y-Coordinate Comparison')
    ax2.legend()
    ax2.grid(True)

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Z Position')
    ax3.set_title('Z-Coordinate Comparison')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.show()


def main():
    dt = 0.001
    n_steps = 1000
    device = torch.device("cpu")

    os.makedirs(experiment_dir, exist_ok=True)

    for engine in ENGINES:
        # Initialize model and sphere for each engine
        model, sphere_idx = create_model(device)

        # Run simulation
        simulate_sphere(model, sphere_idx, dt, n_steps, device, engine)

    # Visualize results
    visualize_trajectories()


if __name__ == "__main__":
    main()