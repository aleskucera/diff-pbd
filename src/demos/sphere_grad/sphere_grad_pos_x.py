import os
import matplotlib.pyplot as plt
import torch
from demos.utils import save_simulation, create_scientific_subplot_plot
from pbd_torch.collision import collide
from pbd_torch.constants import ROT_IDENTITY
from pbd_torch.model import Model
from pbd_torch.model import Quaternion
from pbd_torch.model import Vector3
from pbd_torch.newton_engine import NonSmoothNewtonEngine
from pbd_torch.terrain import create_terrain_from_exr_file
from pbd_torch.xpbd_engine import XPBDEngine
from tqdm import tqdm

PARAM = "pos_x" # Choose from ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"]
EXPERIMENT_NAME = "sphere_bounce_grad_pos_x"

ENGINE = "newton"  # Choose from ["newton", "xpbd"]

AVAILABLE_PARAMS = {
    "pos_x": ("body_q", 0),
    "pos_y": ("body_q", 1),
    "pos_z": ("body_q", 2),
    "vel_x": ("body_qd", 3),
    "vel_y": ("body_qd", 4),
    "vel_z": ("body_qd", 5),
}

assert PARAM in AVAILABLE_PARAMS.keys(), "You must choose parameter form available parameters!"

# Initial state
INITIAL_POSITION = torch.tensor([0.0, 0.0, 4.5]).view(3, 1)
INITIAL_VELOCITY = torch.tensor([5.0, 0.0, -5.0]).view(3, 1)

# Simulation Constants
MAX_CONTACTS_PER_BODY = 36
N_COLLISION_POINTS = 2000
RESTITUTION = 0.8
DYNAMIC_FRICTION = 0.8

# Get the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "..", "..", "data")
experiment_dir = os.path.abspath(os.path.join(data_dir, EXPERIMENT_NAME))

# Output File Names
ANALYTICAL_GRAD_FILE = os.path.join(experiment_dir, "grad_analytical.pt")
NUMERICAL_GRAD_FILE = os.path.join(experiment_dir, "grad_numerical.pt")
LOSS_FILE = os.path.join(experiment_dir, "loss.pt")
PARAM_FILE = os.path.join(experiment_dir, f"{PARAM}.pt")
SIMULATION_FILE = os.path.join(experiment_dir, "simulation.json")

TERRAIN_FILE = os.path.join(data_dir, "blender", "ground4", "textures", "ant01.002_heightmap.exr")
PLOT_FILE = os.path.join(experiment_dir, "gradients_and_loss.png")

def create_model(device: torch.device) -> tuple[Model, int]:
    """Create a sphere model with specified parameters."""
    terrain = create_terrain_from_exr_file(
        heightmap_path=TERRAIN_FILE,
        size_x=40.0,
        size_y=40.0,
        device=device,
    )

    model = Model(max_contacts_per_body=MAX_CONTACTS_PER_BODY, gravity=False)

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

def compute_loss(body_q: torch.Tensor, target_body_q: torch.Tensor) -> torch.Tensor:
    """Compute the loss between the current and target transforms."""
    param_info = AVAILABLE_PARAMS[PARAM]
    param_idx = param_info[1]
    return torch.norm(body_q[param_idx] - target_body_q[param_idx], p=2)
    # return torch.sum((body_q[param_idx] - target_body_q[param_idx]) ** 2)

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
        engine = XPBDEngine(model, pos_iters=10)
    elif ENGINE == "newton":
        engine = NonSmoothNewtonEngine(model, iterations=50)
    else:
        raise ValueError(f"Unknown engine: {ENGINE}")

    control = model.control()
    states = [model.state() for _ in range(n_steps)]

    for i in tqdm(range(n_steps - 1), desc="Simulating"):
        collide(model, states[i], collision_margin=0.0)

        if ENGINE == "xpbd":
            engine.simulate(states[i], states[i + 1], control, dt)
        elif ENGINE == "newton":
            engine.simulate_xitorch(states[i], states[i + 1], control, dt)
        else:
            raise ValueError(f"Unknown engine: {ENGINE}")

    target_transform = states[-1].body_q[sphere_idx].clone()
    print(f"Target transform: {target_transform}")
    print(f"Saving simulation to {SIMULATION_FILE}")
    save_simulation(model, states, SIMULATION_FILE)

    return target_transform

def compute_analytical_gradients(
    model: Model,
    sphere_idx: int,
    target_transform: torch.Tensor,
    dt: float,
    n_steps: int,
) -> torch.Tensor:
    losses = []
    gradients = []

    param_info = AVAILABLE_PARAMS[PARAM]
    model_attribute = param_info[0]
    param_idx = param_info[1]

    initial_param = getattr(model, model_attribute)[sphere_idx, param_idx].item()
    param_range = torch.linspace(initial_param - 1.2, initial_param + 1.2, 51)

    for p in param_range:
        new_model_attribute = getattr(model, model_attribute).clone()
        new_model_attribute[sphere_idx, param_idx] = p
        setattr(model, model_attribute, new_model_attribute.detach().requires_grad_(True))

        if ENGINE == "xpbd":
            engine = XPBDEngine(model, pos_iters=10)
        elif ENGINE == "newton":
            engine = NonSmoothNewtonEngine(model, iterations=50)
        else:
            raise ValueError(f"Unknown engine: {ENGINE}")

        control = model.control()
        states = [model.state() for _ in range(n_steps)]

        for j in range(n_steps - 1):
            collide(model, states[j], collision_margin=0.0)

            if ENGINE == "xpbd":
                engine.simulate(states[j], states[j + 1], control, dt)
            elif ENGINE == "newton":
                engine.simulate_xitorch(states[j], states[j + 1], control, dt)
            else:
                raise ValueError(f"Unknown engine: {ENGINE}")

        loss = compute_loss(states[-1].body_q[sphere_idx], target_transform)
        loss.backward()
        analytical_gradient = getattr(model, model_attribute).grad[sphere_idx, param_idx]

        print(
            f"\nComputing analytical gradient for {PARAM}={p:.2f}\n"
            f"\tLoss: {loss.item():.4f}\n"
            f"\tGradient: {analytical_gradient.item():.4f}"
        )

        gradients.append(analytical_gradient.item())
        losses.append(loss.item())
        if model.body_q.grad is not None:
            model.body_q.grad.zero_()
        if model.body_qd.grad is not None:
            model.body_qd.grad.zero_()

    torch.save(torch.tensor(gradients), ANALYTICAL_GRAD_FILE)
    torch.save(torch.tensor(losses), LOSS_FILE)
    torch.save(param_range, PARAM_FILE)
    return torch.tensor(gradients)

def compute_gradients(
    model: Model,
    sphere_idx: int,
    target_transform: torch.Tensor,
    dt: float,
    n_steps: int,
) -> None:
    """Compute and save both analytical and numerical gradients."""
    compute_analytical_gradients(
        model, sphere_idx, target_transform, dt, n_steps
    )

def visualize_gradients(
    param_file: str = PARAM_FILE,
    loss_file: str = LOSS_FILE,
    analytical_grad_file: str = ANALYTICAL_GRAD_FILE,
) -> None:
    """Visualize loss and gradient data using scientific subplot plot."""
    # Load data
    params = torch.load(param_file)
    loss = torch.load(loss_file)
    analytical_grad = torch.load(analytical_grad_file)

    # Enable LaTeX rendering
    tex_config_setup = {
        'use_tex': True,
        'fonts': 'serif',
        'fontsize': 12,
        'custom_preamble': True,
        'preamble': r'\usepackage{amsmath,amssymb,amsfonts}\usepackage{physics}\usepackage{siunitx}'
    }
    if tex_config_setup['use_tex']:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": tex_config_setup['fonts'],
            "font.size": tex_config_setup['fontsize']
        })
        if tex_config_setup['custom_preamble'] and tex_config_setup['preamble']:
            plt.rcParams["text.latex.preamble"] = tex_config_setup['preamble']

    # Prepare plot data
    plot_data = {
        'Loss': {
            'loss_series': {
                'data': loss.numpy(),
                'label': r'Loss',
                'color': '#1f77b4', # Blue
                'linewidth': 2.0,
                'alpha': 1.0
            }
        },
        'Analytical Gradient': {
            'grad_series': {
                'data': analytical_grad.numpy(),
                'label': r'$\displaystyle\pdv{l}{x_x^0}$',
                'color': '#ff7f0e', # Orange
                'linewidth': 2.0,
                'alpha': 1.0
            }
        }
    }

    y_labels = {
        'Loss': r'Loss (-)',
        'Analytical Gradient': r'$\displaystyle\pdv{l}{x_x^0}$ (-)'
    }

    plot_titles = {
        'Loss': r'Loss vs. Initial $x$-Position',
        'Analytical Gradient': r'Gradient vs. Initial $x$-Position'
    }

    # Plot configuration
    plot_config = {
        'plot_arrangement': 'horizontal',
        'figsize': (10, 5),
        'suptitle': '',
        'x_label': r'Initial $x$-Position $x_x^0$ (m)',
        'y_labels': y_labels,
        'plot_titles': plot_titles,
        'shared_x': True,
        'shared_y': False,
        'dpi': 200,
        'legend': {'show': True, 'location': 'best', 'fontsize': 13},
        'grid': {'show': True, 'linestyle': '--', 'alpha': 0.8},
        'font_sizes': {'axis_label': 13, 'tick_label': 13, 'title': 13, 'suptitle': 14},
        'tight_layout_params': {'rect': [0, 0.02, 1, 0.95]}
    }

    # Create and save the plot
    fig, _ = create_scientific_subplot_plot(
        time_data=params.numpy(),
        plot_data=plot_data,
        config=plot_config,
        save_path=PLOT_FILE
    )
    print(f"Scientific plot saved to {PLOT_FILE}")
    plt.close(fig)

def main():
    dt = 0.01
    n_steps = 100
    device = torch.device("cpu")

    # Initialize model and sphere
    model, sphere_idx = create_model(device)

    # os.makedirs(experiment_dir, exist_ok=True)
    #
    # # Run simulation
    # target_transform = simulate_sphere(model, sphere_idx, dt, n_steps, device)
    #
    # # Compute gradients
    # compute_gradients(model, sphere_idx, target_transform, dt, n_steps)

    # Visualize results
    visualize_gradients()

if __name__ == "__main__":
    main()