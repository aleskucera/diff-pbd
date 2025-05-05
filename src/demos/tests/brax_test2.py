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
       <option timestep="0.01"/>
       <worldbody>
         <body pos="-0.5 0 3.5">
           <joint type="free"/>
           <geom size="0.5" type="sphere"/>
         </body>
         <geom size="40 40 40" type="plane"/>
       </worldbody>
     </mujoco>
"""
)

elasticity = 0.35
# We'll optimize only the x-velocity
ball = ball.replace(elasticity=jnp.array([elasticity] * ball.ngeom))


def simulate(vx, n_steps=200):
    """
    Simulate the ball trajectory with given x-velocity.

    Args:
        vx: Velocity in the x direction
        n_steps: Number of simulation steps

    Returns:
        The entire trajectory and the final state
    """
    # Fixed vertical velocity component
    vz = 1.0  # You can adjust this for a fixed initial upward velocity

    # Initial velocity state with only vx as the parameter
    qd = jnp.array([vx, 0.0, vz, 0.0, 0.0, 0.0], dtype=jnp.float32)

    # Initialize the simulation state
    state = pipeline.init(ball, ball.init_q, qd)

    # Step function to advance the simulation
    def step(state, _):
        next_state = pipeline.step(ball, state, None)
        return next_state, next_state.x.pos[0]

    # Run the simulation
    final_state, positions = jax.lax.scan(step, state, jnp.arange(n_steps))

    return positions, final_state


# Compile the simulation function for better performance
simulate_jit = jit(simulate)


def compute_loss(vx, target_pos):
    """
    Compute loss as the L2 norm between the final position and target position.

    Args:
        vx: Velocity in the x direction
        target_pos: Target 3D position

    Returns:
        Loss value (scalar)
    """
    positions, _ = simulate_jit(vx)

    # Extract the final position (last position from the trajectory)
    final_pos = positions[-1]
    # print(f"Final position: {final_pos}")

    # Compute squared L2 distance
    loss = jnp.sum((final_pos - target_pos) ** 2)

    return loss


# Function to compute gradient of the loss w.r.t. x-velocity
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


def plot_loss_curve(vx_range, target_pos):
    """Plot the loss curve over x-velocity range"""
    loss_values = np.zeros_like(vx_range)
    grad_values = np.zeros_like(vx_range)

    for i, vx in enumerate(vx_range):
        loss_values[i] = compute_loss(vx, target_pos)
        grad_values[i] = compute_grad(vx, target_pos)

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot loss curve
    ax1.plot(vx_range, loss_values, 'b-')
    ax1.set_xlabel('X-Velocity')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs X-Velocity')
    ax1.grid(True)

    # Plot gradient curve
    ax2.plot(vx_range, grad_values, 'r-')
    ax2.set_xlabel('X-Velocity')
    ax2.set_ylabel('Gradient')
    ax2.set_title('Gradient vs X-Velocity')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return loss_values, grad_values


def gradient_descent(init_vx, target_pos, learning_rate=0.1, num_steps=100):
    """Perform gradient descent to find the optimal x-velocity"""
    vx = init_vx
    trajectory = [vx]
    losses = []

    for i in range(num_steps):
        # Compute the loss and gradient
        loss = compute_loss(vx, target_pos)
        grad_vx = compute_grad(vx, target_pos)

        # Update the x-velocity
        vx = vx - learning_rate * grad_vx

        # Store the current vx and loss
        trajectory.append(vx)
        losses.append(loss)

        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss}, Vx: {vx}, Gradient: {grad_vx}")

    return jnp.array(trajectory), jnp.array(losses)


if __name__ == "__main__":
    # Initial x-velocity
    init_vx = 5.0

    # Simulate with initial x-velocity
    positions, _ = simulate_jit(init_vx)
    plot_trajectory(positions, title="Initial Trajectory")

    target_pos = positions[-1]  # Use the last position as the target for optimization
    print(f"Target position for optimization: {target_pos}")

    # Calculate loss and gradient at the initial point
    initial_loss = compute_loss(init_vx, target_pos)
    initial_grad = compute_grad(init_vx, target_pos)

    print(f"Initial loss: {initial_loss}")
    print(f"Initial gradient: {initial_grad}")

    # Plot loss and gradient curves
    vx_range = np.linspace(0.0, 10.0, 100)
    loss_values, grad_values = plot_loss_curve(vx_range, target_pos)

    # Run gradient descent optimization
    trajectory, losses = gradient_descent(init_vx, target_pos,
                                          learning_rate=0.05, num_steps=100)

    # Plot the optimization trajectory on the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(vx_range, loss_values, 'b-', label='Loss Curve')
    plt.scatter(trajectory, [compute_loss(vx, target_pos) for vx in trajectory],
                c='r', s=20, label='Optimization Path')
    plt.scatter(trajectory[0], losses[0], c='g', s=100, label='Start')
    plt.scatter(trajectory[-1], losses[-1], c='m', s=100, label='End')

    plt.xlabel('X-Velocity')
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
    plt.show()

    # Simulate with optimized x-velocity
    optimal_vx = trajectory[-1]
    optimal_positions, _ = simulate_jit(optimal_vx)
    plot_trajectory(optimal_positions, title="Optimized Trajectory")

    print(f"Optimal x-velocity: {optimal_vx}")
    print(f"Final loss: {losses[-1]}")
