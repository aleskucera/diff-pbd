import math
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import brax
import jax
import jax.numpy as jnp
import plotly.graph_objects as go
from brax.io import mjcf
from brax.positional import pipeline
from jax import grad
from jax import jacfwd
from jax import jit

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

ball = mjcf.loads(
    """<mujoco>
       <option timestep="0.01"/>
       <worldbody>
         <body pos="0 0 3">
           <joint type="free"/>
           <geom size="0.5" type="sphere"/>
         </body>
         <geom size="40 40 40" type="plane"/>
       </worldbody>
     </mujoco>
"""
)

elasticity = 0.35
# Instead of a fixed velocity, we'll parameterize both magnitude and angle
ball = ball.replace(elasticity=jnp.array([elasticity] * ball.ngeom))


def simulate(velocity_params, n_steps=400):
    """
    Simulate the ball trajectory with given velocity parameters.

    Args:
        velocity_params: A 2D vector, where velocity_params[0] is magnitude,
                         velocity_params[1] is angle in radians
        n_steps: Number of simulation steps

    Returns:
        The entire trajectory and the final state
    """
    # Extract velocity magnitude and angle
    velocity_mag = velocity_params[0]
    angle = velocity_params[1]

    # Calculate velocity components
    vx = velocity_mag * jnp.cos(angle)
    vz = velocity_mag * jnp.sin(angle)

    # Initial velocity state
    qd = jnp.array([vx, 0, vz, 0, 0, 0], dtype=jnp.float32)

    # Initialize the simulation state
    state = pipeline.init(ball, ball.init_q, qd)

    # Step function to advance the simulation
    def step(state, _):
        next_state = pipeline.step(ball, state, None)
        # Extract the ball position from the state and return it along with the next state
        return next_state, next_state.x.pos[0]

    # Run the simulation
    final_state, positions = jax.lax.scan(step, state, jnp.arange(n_steps))

    return positions, final_state


# Compile the simulation function for better performance
simulate_jit = jit(simulate)


def compute_loss(velocity_params, target_pos):
    """
    Compute loss as the L2 norm between the final position and target position.

    Args:
        velocity_params: A 2D vector [velocity_magnitude, angle]
        target_pos: Target 3D position

    Returns:
        Loss value (scalar)
    """
    positions, final_state = simulate_jit(velocity_params)

    # Extract the final position (last position from the trajectory)
    final_pos = positions[-1]

    # Compute squared L2 distance
    loss = jnp.sum((final_pos - target_pos) ** 2)

    return loss


# Function to compute gradient of the loss w.r.t. velocity parameters
compute_grad = jit(grad(compute_loss))


def plot_trajectory(positions, title="Ball Trajectory"):
    """Plot the trajectory of the ball in 3D"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Convert to numpy for plotting
    positions_np = np.array(positions)

    ax.plot(positions_np[:, 0], positions_np[:, 1], positions_np[:, 2], 'b-')
    ax.scatter(positions_np[0, 0], positions_np[0, 1], positions_np[0, 2], c='g', s=100, label='Start')
    ax.scatter(positions_np[-1, 0], positions_np[-1, 1], positions_np[-1, 2], c='r', s=100, label='End')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

    plt.show()


def plot_loss_surface(velocity_range, angle_range, target_pos):
    """Plot the loss surface over velocity and angle range"""
    v_mesh, a_mesh = np.meshgrid(velocity_range, angle_range)
    loss_values = np.zeros_like(v_mesh)

    for i in range(len(angle_range)):
        for j in range(len(velocity_range)):
            vel_params = jnp.array([velocity_range[j], angle_range[i]])
            loss_values[i, j] = compute_loss(vel_params, target_pos)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(v_mesh, a_mesh, loss_values, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True, alpha=0.8)

    ax.set_xlabel('Velocity Magnitude')
    ax.set_ylabel('Angle (radians)')
    ax.set_zlabel('Loss')
    ax.set_title('Loss Surface')

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    return v_mesh, a_mesh, loss_values


def plot_gradient_field(velocity_range, angle_range, target_pos):
    """Plot the gradient field over velocity and angle range"""
    v_mesh, a_mesh = np.meshgrid(velocity_range, angle_range)
    grad_v = np.zeros_like(v_mesh)
    grad_a = np.zeros_like(a_mesh)

    for i in range(len(angle_range)):
        for j in range(len(velocity_range)):
            vel_params = jnp.array([velocity_range[j], angle_range[i]])
            grads = compute_grad(vel_params, target_pos)
            grad_v[i, j] = grads[0]
            grad_a[i, j] = grads[1]

    plt.figure(figsize=(10, 8))
    plt.streamplot(v_mesh, a_mesh, grad_v, grad_a, density=1.5, color='black', linewidth=1)
    plt.contourf(v_mesh, a_mesh, grad_v ** 2 + grad_a ** 2, cmap='viridis', alpha=0.3)

    plt.xlabel('Velocity Magnitude')
    plt.ylabel('Angle (radians)')
    plt.colorbar(label='Gradient Magnitude')
    plt.title('Gradient Field')
    plt.tight_layout()
    plt.show()

    return grad_v, grad_a


def gradient_descent(init_velocity_params, target_pos, learning_rate=0.1, num_steps=100):
    """Perform gradient descent to find the optimal velocity parameters"""
    velocity_params = init_velocity_params
    trajectory = [velocity_params]
    losses = []

    for i in range(num_steps):
        # Compute the loss and gradient
        loss = compute_loss(velocity_params, target_pos)
        grad = compute_grad(velocity_params, target_pos)

        # Update the velocity parameters
        velocity_params = velocity_params - learning_rate * grad

        # Store the current parameters and loss
        trajectory.append(velocity_params)
        losses.append(loss)

        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss}, Params: {velocity_params}, Grad: {grad}")

    return jnp.array(trajectory), jnp.array(losses)


if __name__ == "__main__":
    # Define target position
    target_pos = jnp.array([15.0, 0.0, 2.0])

    # Initial velocity parameters [magnitude, angle]
    init_velocity_params = jnp.array([6.0, 0.5])

    # Simulate with initial parameters
    positions, _ = simulate_jit(init_velocity_params)
    plot_trajectory(positions, title="Initial Trajectory")

    # Calculate loss and gradient at the initial point
    initial_loss = compute_loss(init_velocity_params, target_pos)
    initial_grad = compute_grad(init_velocity_params, target_pos)

    print(f"Initial loss: {initial_loss}")
    print(f"Initial gradient: {initial_grad}")

    # Plot loss surface and gradient field
    velocity_range = np.linspace(1.0, 10.0, 20)
    angle_range = np.linspace(0.1, 1.5, 20)

    plot_loss_surface(velocity_range, angle_range, target_pos)
    plot_gradient_field(velocity_range, angle_range, target_pos)

    # Run gradient descent optimization
    trajectory, losses = gradient_descent(init_velocity_params, target_pos,
                                          learning_rate=0.05, num_steps=100)

    # Plot optimization trajectory on top of loss surface
    v_mesh, a_mesh, loss_values = plot_loss_surface(velocity_range, angle_range, target_pos)

    plt.figure(figsize=(10, 8))
    plt.contourf(v_mesh, a_mesh, loss_values, levels=20, cmap='viridis')
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', linewidth=2)
    plt.scatter(trajectory[0, 0], trajectory[0, 1], c='g', s=100, label='Start')
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='r', s=100, label='End')

    plt.colorbar(label='Loss')
    plt.xlabel('Velocity Magnitude')
    plt.ylabel('Angle (radians)')
    plt.title('Optimization Path on Loss Surface')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the loss curve during optimization
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs. Iteration')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Simulate with optimized parameters
    optimal_velocity_params = trajectory[-1]
    optimal_positions, _ = simulate_jit(optimal_velocity_params)
    plot_trajectory(optimal_positions, title="Optimized Trajectory")

    print(f"Optimal velocity parameters: {optimal_velocity_params}")
    print(f"Final loss: {losses[-1]}")
