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
from tqdm import tqdm

# Simulation Constants
MAX_CONTACTS_PER_BODY = 36
N_COLLISION_POINTS = 2000
RESTITUTION = 0.8
DYNAMIC_FRICTION = 0.8

INITIAL_POSITION = Vector3(torch.tensor([-0.5, -0.0, 8.5]))
INITIAL_VELOCITY = 5.0
VELOCITY_IDX = 3
VELOCITIES = torch.linspace(0.0, 10.0, 20)

# Get the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "..", "data")
experiment_dir = os.path.join(data_dir, "sphere_throw_grad")

# Output File Names
ANALYTICAL_GRAD_FILE = os.path.join(experiment_dir, "grad_sphere_analytical.pt")
NUMERICAL_GRAD_FILE = os.path.join(experiment_dir, "grad_sphere_numerical.pt")
LOSS_FILE = os.path.join(experiment_dir, "loss_sphere.pt")
VELOCITY_FILE = os.path.join(experiment_dir, "velocity_sphere.pt")
SIMULATION_FILE = os.path.join(experiment_dir, "simulation_sphere.json")

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

    model = Model(max_contacts_per_body=MAX_CONTACTS_PER_BODY)

    sphere_idx = model.add_sphere(
        m=1.0,
        radius=1.0,
        name="sphere",
        pos=INITIAL_POSITION,
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
    model.body_qd[sphere_idx, VELOCITY_IDX] = INITIAL_VELOCITY
    engine = NonSmoothNewtonEngine(model, iterations=150)
    control = model.control()
    states = [model.state() for _ in range(n_steps)]

    for i in tqdm(range(n_steps - 1), desc="Simulating"):
        collide(model, states[i], collision_margin=0.0)
        engine.simulate_xitorch(states[i], states[i + 1], control, dt)

    target_transform = states[-1].body_q[sphere_idx].clone()
    print(f"Target transform: {target_transform}")
    print(f"Saving simulation to {SIMULATION_FILE}")
    save_simulation(model, states, SIMULATION_FILE)

    return target_transform


def compute_analytical_gradients(
    model: Model,
    sphere_idx: int,
    target_transform: torch.Tensor,
    velocities: torch.Tensor,
    dt: float,
    n_steps: int,
) -> torch.Tensor:
    """Compute analytical gradients for sphere velocity."""
    gradients = []
    losses = []

    for velocity in velocities:
        updated_body_qd = model.body_qd.clone()
        updated_body_qd[sphere_idx, VELOCITY_IDX] = velocity
        model.body_qd = updated_body_qd.detach().requires_grad_(True)

        engine = NonSmoothNewtonEngine(model, iterations=150)
        control = model.control()
        states = [model.state() for _ in range(n_steps)]

        for j in range(n_steps - 1):
            collide(model, states[j], collision_margin=0.0)
            engine.simulate_xitorch(states[j], states[j + 1], control, dt)

        loss = torch.norm(states[-1].body_q[sphere_idx] - target_transform)
        loss.backward()
        analytical_gradient = model.body_qd.grad[sphere_idx, 3]

        print(
            f"\nComputing analytical gradient for vx={velocity:.2f}\n"
            f"\tLoss: {loss.item():.4f}\n"
            f"\tGradient: {analytical_gradient.item():.4f}"
        )

        gradients.append(analytical_gradient.item())
        losses.append(loss.item())
        model.body_qd.grad.zero_()

    torch.save(torch.tensor(gradients), ANALYTICAL_GRAD_FILE)
    torch.save(torch.tensor(losses), LOSS_FILE)
    return torch.tensor(gradients)


@torch.no_grad()
def compute_numerical_gradients(
    model: Model,
    sphere_idx: int,
    target_transform: torch.Tensor,
    velocities: torch.Tensor,
    dt: float,
    n_steps: int,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Compute numerical gradients for sphere velocity."""
    gradients = []
    losses = []

    for velocity in velocities:
        engine = NonSmoothNewtonEngine(model, iterations=150)
        control = model.control()

        # Original simulation
        states = [model.state() for _ in range(n_steps)]
        states[0].body_qd[sphere_idx, VELOCITY_IDX] = velocity

        for j in range(n_steps - 1):
            collide(model, states[j], collision_margin=0.0)
            engine.simulate_xitorch(states[j], states[j + 1], control, dt)

        loss_original = torch.norm(states[-1].body_q[sphere_idx] - target_transform)

        # Perturbed simulation
        states_perturbed = [model.state() for _ in range(n_steps)]
        states_perturbed[0].body_qd[sphere_idx, VELOCITY_IDX] = velocity + epsilon

        for j in range(n_steps - 1):
            collide(model, states_perturbed[j], collision_margin=0.0)
            engine.simulate_xitorch(
                states_perturbed[j], states_perturbed[j + 1], control, dt
            )

        loss_perturbed = torch.norm(
            states_perturbed[-1].body_q[sphere_idx] - target_transform
        )
        numerical_gradient = (loss_perturbed - loss_original) / epsilon

        print(
            f"\nComputing numerical gradient for vx={velocity:.2f}\n"
            f"\tOriginal Loss: {loss_original.item():.4f}\n"
            f"\tPerturbed Loss: {loss_perturbed.item():.4f}\n"
            f"\tGradient: {numerical_gradient.item():.4f}"
        )

        gradients.append(numerical_gradient.item())
        losses.append(loss_original.item())

    torch.save(torch.tensor(gradients), NUMERICAL_GRAD_FILE)
    torch.save(torch.tensor(losses), LOSS_FILE)
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
        model, sphere_idx, target_transform, VELOCITIES, dt, n_steps
    )
    compute_numerical_gradients(
        model, sphere_idx, target_transform, VELOCITIES, dt, n_steps
    )
    torch.save(VELOCITIES, VELOCITY_FILE)


def visualize_gradients(
    velocity_file: str = VELOCITY_FILE,
    loss_file: str = LOSS_FILE,
    analytical_grad_file: str = ANALYTICAL_GRAD_FILE,
    numerical_grad_file: str = NUMERICAL_GRAD_FILE,
) -> None:
    """Visualize loss and gradient data."""
    velocities = torch.load(velocity_file)
    loss = torch.load(loss_file)
    analytical_grad = torch.load(analytical_grad_file)
    numerical_grad = torch.load(numerical_grad_file)

    fig, (ax_loss, ax_anal_grad, ax_num_grad) = plt.subplots(3, 1, figsize=(6, 8))

    # Plot Loss
    ax_loss.plot(velocities, loss, label="Loss", color="blue")
    ax_loss.set_title("Loss over Velocities")
    ax_loss.set_xlabel("Velocity (vx)")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True)

    # Plot Analytical Gradients
    ax_anal_grad.plot(
        velocities, analytical_grad, label="Analytical Gradient", color="green"
    )
    ax_anal_grad.set_title("Analytical Gradient")
    ax_anal_grad.set_xlabel("Velocity (vx)")
    ax_anal_grad.set_ylabel("Gradient")
    ax_anal_grad.legend()
    ax_anal_grad.grid(True)

    # Plot Numerical Gradients
    ax_num_grad.plot(
        velocities, numerical_grad, label="Numerical Gradient", color="red"
    )
    ax_num_grad.set_title("Numerical Gradient")
    ax_num_grad.set_xlabel("Velocity (vx)")
    ax_num_grad.set_ylabel("Gradient")
    ax_num_grad.legend()
    ax_num_grad.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_FILE)


def main():
    """Main function to run the sphere simulation and gradient computation."""
    # Simulation parameters
    dt = 0.005
    n_steps = 200
    device = torch.device("cpu")

    # Initialize model and sphere
    model, sphere_idx = create_model(device)

    os.makedirs(experiment_dir, exist_ok=True)

    # Run simulation
    target_transform = simulate_sphere(model, sphere_idx, dt, n_steps, device)

    # Compute gradients
    compute_gradients(model, sphere_idx, target_transform, dt, n_steps)

    # Visualize results
    visualize_gradients()


if __name__ == "__main__":
    main()

