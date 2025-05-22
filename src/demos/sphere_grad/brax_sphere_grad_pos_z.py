import os
import numpy as np
import matplotlib.pyplot as plt
import brax
import jax
import jax.numpy as jnp
from brax.io import mjcf
from brax.generalized import pipeline as generalized_pipeline 
from brax.positional import pipeline as positional_pipeline 
from brax.spring import pipeline as spring_pipeline 
from jax import grad
from jax import jit
from demos.utils import create_scientific_subplot_plot

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

ball = mjcf.loads(
    """<mujoco>
       <option timestep="0.01" gravity="0 0 0"/>
       <worldbody>
         <body pos="0.0 0.0 4.5">
           <joint type="free"/>
           <geom size="1.0" type="sphere"/>
         </body>
         <geom size="40 40 40" type="plane"/>
       </worldbody>
     </mujoco>
"""
)

elasticity = 1.0  # Matches RESTITUTION from sphere_grad_pos_z.py
ball = ball.replace(elasticity=jnp.array([elasticity] * ball.ngeom))

def simulate(z0, n_steps=100):
    """
    Simulate the ball trajectory with given initial z-position.

    Args:
        z0: Initial z-position (height)
        n_steps: Number of simulation steps

    Returns:
        The entire trajectory and the final state
    """
    # Initial velocities from sphere_grad_pos_z.py
    vx = 5.0
    vy = 0.0
    vz = -5.0

    # Initial velocity state
    qd = jnp.array([vx, vy, vz, 0.0, 0.0, 0.0], dtype=jnp.float32)

    # Initial position state: modify z-position, keep x, y, and quaternion
    q = jnp.array([0.0, 0.0, z0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

    # Initialize the simulation state
    state = spring_pipeline.init(ball, q, qd)

    # Step function to advance the simulation
    def step(state, _):
        next_state = spring_pipeline.step(ball, state, None)
        return next_state, next_state.x.pos[0]

    # Run the simulation
    final_state, positions = jax.lax.scan(step, state, jnp.arange(n_steps))

    return positions, final_state

# Compile the simulation function for better performance
simulate_jit = jit(simulate)

def compute_loss(z0, target_z):
    """
    Compute loss as the squared L2 distance between final position and target position.

    Args:
        z0: Initial z-position (height)
        target_pos: Target position [x, y, z]

    Returns:
        Loss value (scalar)
    """
    positions, final_state = simulate_jit(z0)

    # Extract the final position (x, y, z) from the ball's position
    final_z = final_state.x.pos[0, 2]  # Ball is body 0, [x, y, z]
    # print(f"Final z-position: {final_z}")

    # Compute norm distance for all coordinates
    loss = (final_z - target_z) ** 2
    # loss = jnp.sqrt(loss)

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
    """Plot the loss and gradient curves over initial z-position range using scientific subplot plot"""
    loss_values = np.zeros_like(z0_range)
    grad_values = np.zeros_like(z0_range)

    for i, z0 in enumerate(z0_range):
        loss_values[i] = compute_loss(z0, target_z)
        grad_values[i] = compute_grad(z0, target_z)

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
                'data': loss_values,
                'label': r'Loss',
                'color': '#1f77b4', # Blue
                'linewidth': 2.0,
                'alpha': 1.0
            }
        },
        'Gradient': {
            'grad_series': {
                'data': grad_values,
                'label': r'$\displaystyle\pdv{l}{x_z^0}$',
                'color': '#ff7f0e', # Orange
                'linewidth': 2.0,
                'alpha': 1.0
            }
        }
    }

    y_labels = {
        'Loss': r'Loss (-)',
        'Gradient': r'$\displaystyle\pdv{l}{x_z^0}$ (-)'
    }

    plot_titles = {
        'Loss': r'Loss vs. Initial $z$-Position',
        'Gradient': r'Gradient vs. Initial $z$-Position'
    }

    # Plot configuration
    plot_config = {
        'plot_arrangement': 'horizontal',
        'figsize': (10, 5),
        'suptitle': '',
        'x_label': r'Initial $z$-Position $x_z^0$ (m)',
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
    save_path = os.path.join('data', 'sphere_bounce_grad_pos_z_brax', 'gradients_and_loss.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, _ = create_scientific_subplot_plot(
        time_data=z0_range,
        plot_data=plot_data,
        config=plot_config,
        save_path=save_path
    )
    print(f"Scientific plot saved to {save_path}")
    plt.close(fig)

    return loss_values, grad_values

if __name__ == "__main__":
    # Initial z-position
    init_z0 = 4.5  # Matches INITIAL_POSITION[2] from sphere_grad_pos_z.py

    # Simulate with initial z-position
    positions, final_state = simulate_jit(init_z0)
    plot_trajectory(positions, title="Ball Trajectory")

    # Use the final position as the target
    target_z = 2.5  # [x, y, z]
    print(f"Target position for loss computation: {target_z}")

    # Calculate loss and gradient at the initial point
    initial_loss = compute_loss(init_z0, target_z)
    initial_grad = compute_grad(init_z0, target_z)

    print(f"Initial loss: {initial_loss}")
    print(f"Initial gradient: {initial_grad}")

    # Plot loss and gradient curves
    z0_range = np.linspace(init_z0 - 1.5, init_z0 + 1.5, 121)  # Matches param_range (±1.5 around 2.6) from sphere_grad_pos_z.py
    loss_values, grad_values = plot_loss_curve(z0_range, target_z)