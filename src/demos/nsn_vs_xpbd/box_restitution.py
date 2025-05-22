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
from demos.utils import create_scientific_subplot_plot

ENGINES = ["nsn", "xpbd"]
EXPERIMENT_NAME = "box_restitution_comparison"

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
FORCE_FILE = os.path.join(experiment_dir, "force.pt")
TRAJECTORY_FILES = {
    engine: os.path.join(experiment_dir, f"trajectory_{engine}.pt") for engine in ENGINES
}
PLOT_FILE = os.path.join(experiment_dir, "trajectories_comparison.png")

USE_TERRAIN = False
TERRAIN_FILE = os.path.join(data_dir, "blender", "ground4", "textures", "ant01.002_heightmap.exr")
SIMULATION_FIILE = os.path.join(experiment_dir, "simulation.json")


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
        engine_type: str
) -> torch.Tensor:
    model.body_q[box_idx, :3] = INITIAL_POSITION
    model.body_qd[box_idx, 3:] = INITIAL_VELOCITY

    if engine_type == "xpbd":
        engine = XPBDEngine(model, pos_iters=5, contact_compliance=1e-4)
    elif engine_type == "nsn":
        engine = NonSmoothNewtonEngine(model, iterations=150)
    else:
        raise ValueError(f"Unknown engine: {engine_type}")

    control = model.control()
    states = [model.state() for _ in range(n_steps)]

    t = []
    f = []
    q = []

    for i in tqdm(range(n_steps - 1), desc=f"Simulating {engine_type}"):
        collide(model, states[i], collision_margin=0.0)

        if i < 300:
            body_f = 0.0
        else:
            body_f = (i - 300) * 0.01
        body_f = 0.0

        states[i].body_f[box_idx, 3] = body_f

        engine.simulate(states[i], states[i + 1], control, dt)

        t.append(i * dt)
        f.append(body_f)
        q.append(states[i].body_q[box_idx, :3].view(-1).cpu().numpy())

    # Save time and trajectory data
    torch.save(torch.tensor(t), TIME_FILE)
    torch.save(torch.tensor(f), FORCE_FILE)
    torch.save(torch.tensor(q), TRAJECTORY_FILES[engine_type])

    if engine_type == "nsn":
        save_simulation(model, states, SIMULATION_FIILE)

def visualize_trajectories():
    # Load time, force, and trajectory data
    time = torch.load(TIME_FILE).cpu().numpy()
    forces = torch.load(FORCE_FILE).cpu().numpy()

    # Initialize plot data dictionary with the force data
    plot_data = {
        'x': {},
        'y': {},
        'z': {},
        'f': {
            'data': forces,
            'label': r'$f_{x}$',  # TeX notation
            'color': '#2ca02c',
            'linestyle': '-',
            'linewidth': 2.0
        }
    }

    # Load trajectory data for each engine
    for engine in ENGINES:
        trajectory = torch.load(TRAJECTORY_FILES[engine]).cpu().numpy()

        # Add data for each coordinate (x, y, z)
        for i, coord in enumerate(['x', 'y', 'z']):
            # Create a new entry for this engine
            engine_key = f"{engine}"

            # Store the trajectory data for this coordinate and engine
            plot_data[coord][engine_key] = {
                'data': trajectory[:, i],
                'label': f'{engine.upper()}',
                'color': '#1f77b4' if engine == 'nsn' else '#ff7f0e',
                'linestyle': '-' if engine == 'nsn' else '--',
                'linewidth': 2.5
            }

    # Configure the plot with TeX support
    config = {
        'plots_to_show': ['z'],
        'plot_arrangement': 'horizontal',
        'figsize': (5, 5),
        'dpi': 200,
        'suptitle': r'',
        'y_labels': {
            'x': r'$x$ Position (m)',
            'y': r'$y$ Position (m)',
            'z': r'$z$ Position (m)',
            'f': r'$f_{x}$ (N)'
        },
        'plot_titles': {
            'x': r'$x$ - Coordinate',
            'y': r'$y$ - Coordinate',
            'z': r'',
            'f': r'Applied Force Over Time'
        },
        'font_sizes': {
            'title': 14,
            'suptitle': 16,
            'axis_label': 12,
            'tick_label': 12,
            'legend': 12
        },
        'tex': {
            'use_tex': True,  # Enable TeX rendering
            'fonts': 'serif',  # Use serif fonts
            'fontsize': 12,  # Base fontsize for TeX renderer
            'custom_preamble': True,  # Use custom preamble
            'preamble': r'''
                \usepackage{amsmath,amssymb,amsfonts}
                \usepackage{physics}
                \usepackage{siunitx}
            '''  # LaTeX packages for advanced math typesetting
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
    dt = 0.001
    n_steps = 1000
    device = torch.device("cpu")

    os.makedirs(experiment_dir, exist_ok=True)

    for engine in ENGINES:
        # Initialize model and box for each engine
        model, box_idx = create_model(device)

        # Run simulation
        simulate_box(model, box_idx, dt, n_steps, device, engine)

    # Visualize results
    visualize_trajectories()


if __name__ == "__main__":
    main()
