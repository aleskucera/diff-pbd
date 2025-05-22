import os
import time
import numpy as np
import matplotlib.pyplot as plt

import brax
import jax
import jax.numpy as jnp
from brax.io import mjcf
from brax.positional import pipeline
from jax import grad
from jax import jit

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

ball = mjcf.loads(
    """<mujoco>
       <option timestep="0.01" gravity="0 0 0"/>
       <worldbody>
         <body pos="0.0 0 1.0">
           <joint type="free"/>
           <geom size="0.001" type="sphere"/>
         </body>
         <geom size="40 40 40" type="plane"/>
       </worldbody>
     </mujoco>
"""
)

elasticity = 1.0
ball = ball.replace(elasticity=jnp.array([elasticity] * ball.ngeom))


def simulate(z0, n_steps=200):
    """
    Simulate the ball trajectory with given initial z-position.

    Args:
        z0: Initial z-position (height)
        n_steps: Number of simulation steps

    Returns:
        The entire trajectory and the final state
    """
    # Fixed initial velocities
    vx = 1.0  # Fixed x-velocity (as in original init_vx)
    vz = -1.0  # Fixed vertical velocity

    # Initial velocity state
    qd = jnp.array([vx, 0.0, vz, 0.0, 0.0, 0.0], dtype=jnp.float32)

    # Initial position state: modify z-position, keep x, y, and quaternion
    # Default position from MJCF is [-0.5, 0, 3.5], quaternion [1, 0, 0, 0]
    q = jnp.array([0.0, 0.0, z0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

    # Initialize the simulation state
    state = pipeline.init(ball, q, qd)

    # Step function to advance the simulation
    def step(state, _):
        next_state = pipeline.step(ball, state, None)
        return next_state, next_state.x.pos[0]

    # Run the simulation
    final_state, positions = jax.lax.scan(step, state, jnp.arange(n_steps))

    return positions, final_state


# Compile the simulation function for better performance
simulate_jit = jit(simulate)


def compute_loss(z0, target_z):
    """
    Compute loss as the squared difference between final z-position and target z-position.

    Args:
        z0: Initial z-position (height)
        target_z: Target z-position (height)

    Returns:
        Loss value (scalar)
    """
    positions, final_state = simulate_jit(z0)

    # Extract the final z-position from the ball's position
    final_z = final_state.x.pos[0, 2]  # Ball is body 0, z-coordinate

    # Compute squared L2 distance for z-coordinate
    loss = (final_z - target_z) ** 2

    return loss


# Function to compute gradient of the loss w.r.t. z-position
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


def plot_loss_curve(z0_range, target_z):
    """Plot the loss curve over initial z-position range"""
    loss_values = np.zeros_like(z0_range)
    grad_values = np.zeros_like(z0_range)

    for i, z0 in enumerate(z0_range):
        loss_values[i] = compute_loss(z0, target_z)
        grad_values[i] = compute_grad(z0, target_z)

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot loss curve
    ax1.plot(z0_range, loss_values, 'b-')
    ax1.set_xlabel('Initial Z-Position')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Initial Z-Position')
    ax1.grid(True)

    # Plot gradient curve
    ax2.plot(z0_range, grad_values, 'r-')
    ax2.set_xlabel('Initial Z-Position')
    ax2.set_ylabel('Gradient')
    ax2.set_title('Gradient vs Initial Z-Position')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return loss_values, grad_values


def gradient_descent(init_z0, target_z, learning_rate=0.1, num_steps=100):
    """Perform gradient descent to find the optimal initial z-position"""
    z0 = init_z0
    trajectory = [z0]
    losses = []

    for i in range(num_steps):
        # Compute the loss and gradient
        loss = compute_loss(z0, target_z)
        grad_z0 = compute_grad(z0, target_z)

        # Update the z-position
        z0 = z0 - learning_rate * grad_z0

        # Store the current z0 and loss
        trajectory.append(z0)
        losses.append(loss)

        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss}, Z0: {z0}, Gradient: {grad_z0}")

    return jnp.array(trajectory), jnp.array(losses)


if __name__ == "__main__":
    # Initial z-position
    init_z0 = 1.0

    # Simulate with initial z-position
    positions, final_state = simulate_jit(init_z0)
    plot_trajectory(positions, title="Initial Trajectory")

    # Use the final z-position as the target for optimization
    target_z = final_state.x.pos[0, 2]
    print(f"Target z-position for optimization: {target_z}")

    # Calculate loss and gradient at the initial point
    initial_loss = compute_loss(init_z0, target_z)
    initial_grad = compute_grad(init_z0, target_z)

    print(f"Initial loss: {initial_loss}")
    print(f"Initial gradient: {initial_grad}")

    # Plot loss and gradient curves
    z0_range = np.linspace(0.51, 1.51, 20)  # Range around initial z=position
    loss_values, grad_values = plot_loss_curve(z0_range, target_z)

    # Run gradient descent optimization
    trajectory, losses = gradient_descent(0.8, target_z,
                                         learning_rate=0.05, num_steps=10)

    # Plot the optimization trajectory on the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(z0_range, loss_values, 'b-', label='Loss Curve')
    plt.scatter(trajectory, [compute_loss(z0, target_z) for z0 in trajectory],
                c='r', s=20, label='Optimization Path')
    plt.scatter(trajectory[0], losses[0], c='g', s=100, label='Start')
    plt.scatter(trajectory[-1], losses[-1], c='m', s=100, label='End')

    plt.xlabel('Initial Z-Position')
    plt.ylabel('Loss')
    plt.title('Optimization Path on Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the loss curve during optimization
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs. Iteration')
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.show()

    # Simulate with optimized z-position
    optimal_z0 = trajectory[-1]
    optimal_positions, _ = simulate_jit(optimal_z0)
    plot_trajectory(optimal_positions, title="Optimized Trajectory")

    print(f"Optimal initial z-position: {optimal_z0}")
    print(f"Final loss: {losses[-1]}")