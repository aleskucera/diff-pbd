import os
import numpy as np
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

# Experiment configuration
EXPERIMENT_NAME = "restitution_comparison"
COMPLIANCE_VALUES = [1e-3, 1e-4, 1e-6, 1e-8]
DEFAULT_COMPLIANCE = 1e-4

# Initial state
INITIAL_POSITION = torch.tensor([0.0, 0.0, 0.51]).view(3, 1)
INITIAL_VELOCITY = torch.tensor([0.0, 0.0, 0.0]).view(3, 1)

# Simulation Constants
MAX_CONTACTS_PER_BODY = 200
N_COLLISION_POINTS = 1000
RESTITUTION = 0.0
DYNAMIC_FRICTION = 1.0

# Get the current working directory and set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "..", "..", "data")
experiment_dir = os.path.abspath(os.path.join(data_dir, EXPERIMENT_NAME))

# Output file names
TIME_FILE = os.path.join(experiment_dir, "time.pt")
STANDARD_TRAJECTORY_FILES = {
    "nsn": os.path.join(experiment_dir, "trajectory_nsn.pt"),
    "xpbd": os.path.join(experiment_dir, f"trajectory_xpbd.pt")
}
COMPLIANCE_TRAJECTORY_FILES = {
    f"compliance_{compliance:.0e}": os.path.join(
        experiment_dir, f"trajectory_compliance_{compliance:.0e}.pt"
    )
    for compliance in COMPLIANCE_VALUES
}
PLOT_FILE = os.path.join(experiment_dir, "combined_comparison.png")

# Terrain settings
USE_TERRAIN = False
TERRAIN_FILE = os.path.join(data_dir, "blender", "ground4", "textures", "ant01.002_heightmap.exr")
SIMULATION_FILE = os.path.join(experiment_dir, "simulation.json")


def create_model(device: torch.device) -> tuple[Model, int]:
    """Create a box model for simulation."""
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

    box_idx = model.add_box(
        m=1.0,
        hx=0.5,
        hy=0.5,
        hz=0.5,
        name="box",
        pos=Vector3(INITIAL_POSITION.view(-1)),
        rot=Quaternion(ROT_IDENTITY),
        restitution=RESTITUTION,
        dynamic_friction=DYNAMIC_FRICTION,
        n_collision_points=N_COLLISION_POINTS,
    )
    return model, box_idx


def simulate_standard_comparison(
        model: Model,
        box_idx: int,
        dt: float,
        n_steps: int,
        device: torch.device,
        engine_type: str
) -> None:
    """Run simulation with either NSN or XPBD (with default compliance)."""
    model.body_q[box_idx, :3] = INITIAL_POSITION
    model.body_qd[box_idx, 3:] = INITIAL_VELOCITY

    if engine_type == "xpbd":
        engine = XPBDEngine(model, pos_iters=5, contact_compliance=DEFAULT_COMPLIANCE)
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

        # No external forces applied
        states[i].body_f[box_idx, :] = 0.0

        engine.simulate(states[i], states[i + 1], control, dt)

        t.append(i * dt)
        q.append(states[i].body_q[box_idx, :3].view(-1).cpu().numpy())

    # Save time and trajectory data
    # Convert lists to numpy arrays before creating tensors to avoid the warning
    t_array = np.array(t)
    q_array = np.array(q)

    torch.save(torch.tensor(t_array), TIME_FILE)
    torch.save(torch.tensor(q_array), STANDARD_TRAJECTORY_FILES[engine_type])

    # Save the simulation for NSN
    if engine_type == "nsn":
        save_simulation(model, states, SIMULATION_FILE)


def simulate_compliance_comparison(
        model: Model,
        box_idx: int,
        dt: float,
        n_steps: int,
        device: torch.device,
        compliance: float
) -> None:
    """Run simulation with XPBD using different compliance values."""
    model.body_q[box_idx, :3] = INITIAL_POSITION
    model.body_qd[box_idx, 3:] = INITIAL_VELOCITY

    # Create XPBD engine with the specified compliance
    engine = XPBDEngine(model, pos_iters=5, contact_compliance=compliance)

    control = model.control()
    states = [model.state() for _ in range(n_steps)]

    t = []
    q = []

    for i in tqdm(range(n_steps - 1), desc=f"Simulating XPBD with α={compliance:.0e}"):
        collide(model, states[i], collision_margin=0.0)

        # No external forces applied
        states[i].body_f[box_idx, :] = 0.0

        engine.simulate(states[i], states[i + 1], control, dt)

        t.append(i * dt)
        q.append(states[i].body_q[box_idx, :3].view(-1).cpu().numpy())

    # Save time data if not already saved
    if not os.path.exists(TIME_FILE):
        # Convert list to numpy array before creating tensor
        t_array = np.array(t)
        torch.save(torch.tensor(t_array), TIME_FILE)

    # Save trajectory for this compliance value
    # Convert list to numpy array before creating tensor
    q_array = np.array(q)
    compliance_key = f"compliance_{compliance:.0e}"
    torch.save(torch.tensor(q_array), COMPLIANCE_TRAJECTORY_FILES[compliance_key])


def create_combined_plot():
    """Create a combined figure with two subplots, without using LaTeX."""
    # Load time data
    time = torch.load(TIME_FILE).cpu().numpy()

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=200)

    # Disable LaTeX rendering to avoid the error
    plt.rcParams['text.usetex'] = False

    # Load and plot NSN vs XPBD comparison in the first subplot
    for engine, style in zip(["nsn", "xpbd"], ['-', '--']):
        trajectory = torch.load(STANDARD_TRAJECTORY_FILES[engine]).cpu().numpy()
        z_position = trajectory[:, 2]  # Get z-coordinate
        label = "NSN" if engine == "nsn" else "XPBD"
        color = "#1f77b4" if engine == "nsn" else "#ff7f0e"
        ax1.plot(time, z_position, label=label, color=color, linestyle=style, linewidth=2)

    # Configure first subplot
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("z Position (m)")
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='best')
    # ax1.set_ylim(0.495, 0.51)

    # Load and plot XPBD compliance comparison in the second subplot
    colors = plt.cm.viridis(np.linspace(0, 1, len(COMPLIANCE_VALUES)))
    line_styles = ['-', '--', '-.', ':']

    for i, compliance in enumerate(COMPLIANCE_VALUES):
        compliance_key = f"compliance_{compliance:.0e}"
        trajectory = torch.load(COMPLIANCE_TRAJECTORY_FILES[compliance_key]).cpu().numpy()
        z_position = trajectory[:, 2]  # Get z-coordinate
        label = f"α = {compliance:.0e}"
        ax2.plot(
            time,
            z_position,
            label=label,
            color=colors[i],
            linestyle=line_styles[i % 4],
            linewidth=2
        )

    # Configure second subplot
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("z Position (m)")
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='best')
    # ax2.set_ylim(0.49, 0.51)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(PLOT_FILE, bbox_inches='tight')
    plt.close()


def main():
    dt = 0.001
    n_steps = 1000
    device = torch.device("cpu")

    # Create output directory
    os.makedirs(experiment_dir, exist_ok=True)

    # Run standard comparison: NSN vs XPBD
    for engine in ["nsn", "xpbd"]:
        model, box_idx = create_model(device)
        simulate_standard_comparison(model, box_idx, dt, n_steps, device, engine)

    # Run XPBD compliance comparison
    for compliance in COMPLIANCE_VALUES:
        model, box_idx = create_model(device)
        simulate_compliance_comparison(model, box_idx, dt, n_steps, device, compliance)

    # Create the combined visualization
    create_combined_plot()

    print(f"Combined comparison plot saved to {PLOT_FILE}")


if __name__ == "__main__":
    main()
