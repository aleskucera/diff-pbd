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
from pbd_torch.terrain import create_terrain_from_exr_file
from pbd_torch.xpbd_engine import XPBDEngine
from tqdm import tqdm
from demos.utils import create_scientific_subplot_plot

# Define compliance values to compare
COMPLIANCE_VALUES = [1e-3, 1e-4, 1e-6, 1e-8]
EXPERIMENT_NAME = "xpbd_compliance_comparison"

# Initial state
INITIAL_POSITION = torch.tensor([0.0, 0.0, 0.51]).view(3, 1)
INITIAL_VELOCITY = torch.tensor([0.0, 0.0, 0.0]).view(3, 1)

# Simulation Constants
MAX_CONTACTS_PER_BODY = 200
N_COLLISION_POINTS = 1000
RESTITUTION = 0.0
DYNAMIC_FRICTION = 1.0

# Get the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "..", "..", "data")
experiment_dir = os.path.abspath(os.path.join(data_dir, EXPERIMENT_NAME))

# Output File Names
TIME_FILE = os.path.join(experiment_dir, "time.pt")
TRAJECTORY_FILES = {
    f"compliance_{compliance:.0e}": os.path.join(experiment_dir, f"trajectory_compliance_{compliance:.0e}.pt")
    for compliance in COMPLIANCE_VALUES
}
PLOT_FILE = os.path.join(experiment_dir, "compliance_comparison.png")

USE_TERRAIN = False
TERRAIN_FILE = os.path.join(data_dir, "blender", "ground4", "textures", "ant01.002_heightmap.exr")
SIMULATION_FILE = os.path.join(experiment_dir, "simulation.json")


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


def simulate_box(
        model: Model,
        box_idx: int,
        dt: float,
        n_steps: int,
        device: torch.device,
        compliance: float
) -> None:
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

        # Apply force in x direction
        states[i].body_f[box_idx, 3] = body_f

        engine.simulate(states[i], states[i + 1], control, dt)

        t.append(i * dt)
        q.append(states[i].body_q[box_idx, :3].view(-1).cpu().numpy())

    # Save time and trajectory data
    torch.save(torch.tensor(t), TIME_FILE)
    torch.save(torch.tensor(f), FORCE_FILE)

    # Save trajectory for this compliance value
    compliance_key = f"compliance_{compliance:.0e}"
    torch.save(torch.tensor(q), TRAJECTORY_FILES[compliance_key])


def visualize_trajectories():
    # Load time and force data
    time = torch.load(TIME_FILE).cpu().numpy()

    # Initialize plot data dictionary
    plot_data = {
        'x': {},  # x position
        'y': {},  # y position
        'z': {},  # z position
    }

    # Color map for different compliance values
    colors = plt.cm.viridis(np.linspace(0, 1, len(COMPLIANCE_VALUES)))

    # Load trajectory data for each compliance value
    for i, compliance in enumerate(COMPLIANCE_VALUES):
        compliance_key = f"compliance_{compliance:.0e}"
        trajectory = torch.load(TRAJECTORY_FILES[compliance_key]).cpu().numpy()

        # Add data for each coordinate (x, y, z)
        for j, coord in enumerate(['x', 'y', 'z']):
            # Format the label with alpha value in scientific notation
            alpha_label = f"{compliance:.0e}"

            # Store the trajectory data for this coordinate and compliance
            plot_data[coord][compliance_key] = {
                'data': trajectory[:, j],
                'label': r'$\alpha = ' + alpha_label + '$',
                'color': colors[i],
                'linestyle': ['-', '--', '-.', ':'][i % 4],  # Cycle through line styles
                'linewidth': 2.5
            }

    # Configure the plot with TeX support
    config = {
        'plots_to_show': ['z'],  # Show x, z and force
        'plot_arrangement': 'horizontal',
        'figsize': (5, 5),
        'dpi': 200,
        # 'suptitle': r'Effect of Compliance ($\alpha$) on XPBD Simulation',
        'y_labels': {
            'x': r'$x$ Position (m)',
            'y': r'$y$ Position (m)',
            'z': r'$z$ Position (m)',
            'f': r'$f_{x}$ (N)'
        },
        'plot_titles': {
            'x': r'$x$ Coordinate over Time',
            'y': r'$y$ Coordinate over Time',
            'z': r'',
            'f': r'Applied Force'
        },
        # 'y_limits': {
        #     'z': [0, 10.0],  # Limit z-axis to make differences more visible
        # },
        'font_sizes': {
            'title': 14,
            'suptitle': 16,
            'axis_label': 12,
            'tick_label': 12,
            'legend': 12
        },
        'tex': {
            'use_tex': True,
            'fonts': 'serif',
            'fontsize': 12,
            'custom_preamble': True,
            'preamble': r'''
                \usepackage{amsmath,amssymb,amsfonts}
                \usepackage{physics}
                \usepackage{siunitx}
            '''
        },
        'grid': {
            'linestyle': '--',
            'alpha': 0.8
        },
        'legend': {
            'location': 'best'
        }
    }

    # Create the plot
    _, _ = create_scientific_subplot_plot(time, plot_data, config, save_path=PLOT_FILE)


def main():
    # dt = 0.001
    # n_steps = 1000
    # device = torch.device("cpu")
    #
    # os.makedirs(experiment_dir, exist_ok=True)
    #
    # # Run simulations for each compliance value
    # for compliance in COMPLIANCE_VALUES:
    #     # Initialize model and box
    #     model, box_idx = create_model(device)
    #
    #     # Run simulation with this compliance value
    #     simulate_box(model, box_idx, dt, n_steps, device, compliance)

    # Visualize results
    visualize_trajectories()


if __name__ == "__main__":
    main()
