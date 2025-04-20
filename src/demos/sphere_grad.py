import os

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
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

def simulate_sphere():
    dt = 0.01
    n_steps = 200
    device = torch.device("cpu")
    output_file = os.path.join("simulation", "sphere_grad.json")

    terrain = create_terrain_from_exr_file(
        heightmap_path="/home/kuceral4/school/diff-pbd/data/blender/ground4/textures/ant01.002_heightmap.exr",
        size_x=40.0,
        size_y=40.0,
        device=device,
    )

    model = Model(max_contacts_per_body=16)

    sphere = model.add_sphere(
        m=1.0,
        radius=1.0,
        name="sphere",
        pos=Vector3(torch.tensor([-0.5, -1.0, 4.5])),
        rot=Quaternion(ROT_IDENTITY),
        restitution=0.8,
        dynamic_friction=0.8,
        n_collision_points=4000,
    )

    target_trans = torch.tensor([[-7.5102e-01],
                                 [-9.6616e-01],
                                 [ 3.8366e+00],
                                 [ 9.9198e-01],
                                 [-1.6892e-02],
                                 [-1.2525e-01],
                                 [ 6.3300e-04]])


    model.body_qd[sphere, 3:] = torch.tensor([0.0, 0.0, -0.5]).view(3, 1)

    engine = NonSmoothNewtonEngine(model, iterations=150)

    control = model.control()
    states = [model.state() for _ in range(n_steps)]

    # Simulate the model
    for i in tqdm(range(n_steps - 1), desc="Simulating"):
        collide(model, states[i], collision_margin=0.0)
        engine.simulate_xitorch(states[i], states[i + 1], control, dt)

    loss = torch.norm(states[-1].body_q[sphere] - target_trans)

    print(f"Last transform: {states[-1].body_q[sphere]}")
    print(f"Loss: {loss.item()}")

    print(f"Saving simulation to {output_file}")
    save_simulation(model, states, output_file)

def compute_gradients():
    dt = 0.01
    n_steps = 200
    device = torch.device("cpu")
    output_file = os.path.join("simulation", "sphere_grad.json")

    terrain = create_terrain_from_exr_file(
        heightmap_path="/home/kuceral4/school/diff-pbd/data/blender/ground4/textures/ant01.002_heightmap.exr",
        size_x=40.0,
        size_y=40.0,
        device=device,
    )

    model = Model(max_contacts_per_body=16)

    sphere = model.add_sphere(
        m=1.0,
        radius=1.0,
        name="sphere",
        pos=Vector3(torch.tensor([-0.5, -1.0, 4.5])),
        rot=Quaternion(ROT_IDENTITY),
        restitution=0.8,
        dynamic_friction=0.8,
        n_collision_points=400,
    )

    target_trans = torch.tensor([[-7.5102e-01],
        [-9.6616e-01],
        [ 3.8366e+00],
        [ 9.9198e-01],
        [-1.6892e-02],
        [-1.2525e-01],
        [ 6.3300e-04]])

    vz = torch.linspace(-1.0, 1.0, 20)
    grads = get_analytical_dvz(
        model,
        sphere,
        vz,
        dt,
        n_steps,
        target_trans)

    torch.save(vz, "vz.pt")

def get_analytical_dvz(
        model: Model,
        sphere: int,
        vz: torch.Tensor,
        dt: float,
        n_steps: int,
        target_trans: torch.Tensor):
    grads = []
    losses = []

    for i, _vz in enumerate(vz):
        updated_body_qd = model.body_qd.clone()
        updated_body_qd[sphere, 5] = _vz
        model.body_qd = updated_body_qd.detach().requires_grad_(True)

        engine = NonSmoothNewtonEngine(model, iterations=150)

        control = model.control()
        states = [model.state() for _ in range(n_steps)]

        # Simulate the model
        for j in range(n_steps - 1):
            collide(model, states[j], collision_margin=0.0)
            engine.simulate_xitorch(states[j], states[j + 1], control, dt)

        # Compute the loss
        loss = torch.norm(states[-1].body_q[sphere][2] - target_trans[2])

        # Compute the gradient
        loss.backward()

        # Get the gradient of the velocity
        anal_grad = model.body_qd.grad[sphere, 5]

        print(f"\nComputing analytical gradient for vz={_vz} \n"
              f"\tLoss: {loss.item():.4f} \n"
              f"\tGradient: {anal_grad.item():.4f}")

        grads.append(anal_grad.item())
        losses.append(loss.item())
        model.body_qd.grad.zero_()

    torch.save(grads, "grad_sphere_analytical.pt")
    torch.save(losses, "loss_sphere_analytical.pt")
    return torch.tensor(grads)

@torch.no_grad()
def get_numerical_dvz(
        model: Model,
        sphere: int,
        vz: torch.Tensor,
        dt: float,
        n_steps: int,
        target_trans: torch.Tensor,
        epsilon: float = 1e-4):
    grads = []
    losses = []

    for i, _vz in enumerate(vz):
        # Simulate with original velocity
        model.body_qd[sphere, 5] = _vz

        engine = NonSmoothNewtonEngine(model, iterations=150)

        control = model.control()
        states = [model.state() for _ in range(n_steps)]

        for j in range(n_steps - 1):
            collide(model, states[j], collision_margin=0.0)
            engine.simulate_xitorch(states[j], states[j + 1], control, dt)

        loss_original = torch.norm(states[-1].body_q[sphere, 2] - target_trans[2])

        # Simulate with perturbed velocity
        model.body_qd[sphere, 5] += epsilon

        states_perturbed = [model.state() for _ in range(n_steps)]

        for j in range(n_steps - 1):
            collide(model, states_perturbed[j], collision_margin=0.0)
            engine.simulate_xitorch(states_perturbed[j], states_perturbed[j + 1], control, dt)

        loss_perturbed = torch.norm(states_perturbed[-1].body_q[sphere, 2] - target_trans[2])

        # Numerical gradient
        numerical_grad = (loss_perturbed - loss_original) / epsilon

        print(f"\nComputing numerical gradient for vz={_vz} \n"
              f"\tOriginal Loss: {loss_original.item()} \n"
              f"\tPerturbed Loss: {loss_perturbed.item()} \n"
              f"\tGradient: {numerical_grad.item()}")

        grads.append(numerical_grad.item())
        losses.append(loss_original.item())

    torch.save(grads, "grad_sphere_numerical.pt")
    torch.save(losses, "loss_sphere_numerical.pt")

    return torch.tensor(grads)

def visualize_gradients(anal_grad_file="loss_sphere_analytical.pt",
                        num_grad_file="loss_sphere_numerical.pt",
                        x_file="vz.pt"):
    # Load the gradients and velocities
    anal_grads = torch.load(anal_grad_file)
    num_grads = torch.load(num_grad_file)
    x = torch.load(x_file).cpu().numpy()

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, anal_grads, label="Analytical Gradient", marker="o", linestyle="-")
    plt.plot(x, num_grads, label="Numerical Gradient", marker="x", linestyle="--", alpha=0.7)
    # Add labels, title, and legend
    plt.xlabel("Velocity (vz)")
    plt.ylabel("Gradient")
    plt.title("Gradient vs Velocity for Sphere")
    plt.legend()
    plt.grid()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # simulate_sphere()
    # compute_gradients()
    visualize_gradients()