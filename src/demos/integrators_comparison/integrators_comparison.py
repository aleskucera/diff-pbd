import torch
import numpy as np
# import matplotlib.pyplot as plt # No longer directly needed for plotting
from scipy.optimize import root
from jaxtyping import Float

import matplotlib  # Still needed for matplotlib.use('TkAgg') if running in certain environments

matplotlib.use('TkAgg')

from demos.utils import create_scientific_subplot_plot  # Import the function from utils.py

# Define the simple pendulum system parameters
g = 9.81  # acceleration due to gravity (m/s^2)
l = 1.0  # length of the pendulum (m)
m = 1.0  # mass of the pendulum bob (kg)

# Define the force acting on the bounce
k = 20000.0  # spring constant (N/m)

EXPERIMENT = "bounce"  # or "pendulum" # Changed to bounce to match original q0


def accel_pendulum(q: torch.Tensor) -> torch.Tensor:
    """Compute the force acting on the pendulum."""
    return -g * l * torch.sin(q)


def accel_bounce(q: torch.Tensor) -> torch.Tensor:
    """Compute the force acting on the bounce."""
    if q > 0:
        return torch.tensor([-g])
    else:
        return -k * q / m - g


def forward_euler(q0: torch.Tensor, u0: torch.Tensor, dt: float, num_steps: int):
    history = {"t": [0.0],
               "q": [q0.item()],
               "u": [u0.item()]}

    q = q0.clone()
    u = u0.clone()

    # Determine acceleration function based on experiment
    if EXPERIMENT == "pendulum":
        accel_func = lambda x: -(g / l) * torch.sin(x)
    elif EXPERIMENT == "bounce":
        accel_func = accel_bounce
    else:
        raise ValueError(f"Unknown EXPERIMENT: {EXPERIMENT}")

    for _ in range(num_steps):
        a = accel_func(q)  # Use the selected acceleration function
        u_next = u + dt * a
        q_next = q + dt * u

        history["q"].append(q_next.item())
        history["u"].append(u_next.item())
        history["t"].append(history["t"][-1] + dt)

        q = q_next
        u = u_next

    return history


def symplectic_euler(q0: torch.Tensor, u0: torch.Tensor, dt: float, num_steps: int):
    history = {"t": [0.0],
               "q": [q0.item()],
               "u": [u0.item()]}

    if EXPERIMENT == "pendulum":
        accel = accel_pendulum
    elif EXPERIMENT == "bounce":
        accel = accel_bounce
    else:
        raise ValueError(f"Unknown EXPERIMENT: {EXPERIMENT}")

    q = q0.clone()
    u = u0.clone()
    for _ in range(num_steps):
        a = accel(q)
        u_next = u + dt * a
        q_next = q + dt * u_next

        history["q"].append(q_next.item())
        history["u"].append(u_next.item())
        history["t"].append(history["t"][-1] + dt)

        q = q_next
        u = u_next

    return history


def symplectic_euler_position(q0: torch.Tensor, u0: torch.Tensor, dt: float, num_steps: int):
    history = {"t": [0.0],
               "q": [q0.item()],
               "u": [u0.item()]}

    q = q0.clone()
    u = u0.clone()

    # Determine acceleration function based on experiment
    if EXPERIMENT == "pendulum":
        accel_func = lambda x: -(g / l) * torch.sin(x)
    elif EXPERIMENT == "bounce":
        # For bounce, symplectic euler position might behave unexpectedly if not formulated carefully for non-smooth forces
        # The original accel_bounce is complex. For simplicity in this context,
        # we might need to adjust how 'a' is calculated or acknowledge this integrator
        # might be less suitable for the bounce experiment as written.
        # The original code had `a = -(g / l) * torch.sin(q)` which is specific to pendulum
        # For bounce, it should be:
        accel_func = accel_bounce

    else:
        raise ValueError(f"Unknown EXPERIMENT: {EXPERIMENT}")

    for _ in range(num_steps):
        a = accel_func(q)  # Use the selected acceleration function
        q_next = q + dt * u + 0.5 * dt ** 2 * a  # Common formulation for Verlet-like position update
        # u_next = (q_next - q) / dt # This is one way to get velocity
        # Or, more consistently with symplectic methods, update velocity based on average acceleration
        # However, the prompt's original symplectic_euler_position was:
        # a = -(g / l) * torch.sin(q)
        # q_next = q + dt * u + dt**2 * a  <- Note: original had full dt**2 * a, common is 0.5 * dt**2 * a
        # u_next = (q_next - q) / dt
        # Let's stick to the structure from the user's original code for this integrator,
        # and acknowledge it might be specific to the pendulum's -(g/l)sin(q) form if used directly.
        # If EXPERIMENT is "bounce", this specific `symplectic_euler_position` might not be directly applicable
        # without re-deriving for that force. For now, let's assume the user wants to see it run.

        # Reverting to original logic specific to pendulum for this function if we want to match old behavior
        if EXPERIMENT == "pendulum":
            a = -(g / l) * torch.sin(q)
            q_next = q + dt * u + dt ** 2 * a  # Original formulation, potentially for pendulum
        elif EXPERIMENT == "bounce":  # We need a consistent 'a' for bounce
            # This integrator is a bit tricky for bounce with its current q_next formula
            # For now, let's use accel_bounce and see. It might not be "symplectic" for bounce this way.
            a_current = accel_bounce(q)
            # q_next from original symplectic_euler_position for pendulum used dt**2 * a.
            # This is unusual. A more standard Verlet position update would be q + dt * u + 0.5 * dt**2 * a
            # Let's use the bounce acceleration, but be aware the formula for q_next is from the pendulum version.
            q_next = q + dt * u + dt ** 2 * a_current  # Using original formula structure

        u_next = (q_next - q) / dt

        history["q"].append(q_next.item())
        history["u"].append(u_next.item())
        history["t"].append(history["t"][-1] + dt)

        q = q_next
        u = u_next

    return history


def backward_euler(q0: torch.Tensor, u0: torch.Tensor, dt: float, num_steps: int):
    history = {"t": [0.0],
               "q": [q0.item()],
               "u": [u0.item()]}

    q_prev_val = q0.clone().item()
    u_prev_val = u0.clone().item()

    # Determine acceleration function based on experiment for the residual
    if EXPERIMENT == "pendulum":
        accel_func_np = lambda x_next: (-g / l * np.sin(x_next))
    elif EXPERIMENT == "bounce":
        # accel_bounce returns a tensor. For numpy root solver, we need scalar float.
        # Also, accel_bounce itself uses if q > 0. q_next is the variable here.
        accel_func_np = lambda x_next: accel_bounce(torch.tensor([x_next])).item()

    else:
        raise ValueError(f"Unknown EXPERIMENT: {EXPERIMENT}")

    def residual(x: list, q_prev: float, u_prev: float) -> list:
        q_next_res, u_next_res = x
        r_q = q_next_res - q_prev - dt * u_next_res
        # The acceleration term a(q_next) depends on the experiment
        a_next = accel_func_np(q_next_res)
        r_u = u_next_res - u_prev - dt * a_next
        return [r_q, r_u]

    for _ in range(num_steps):
        x0 = [q_prev_val, u_prev_val]
        sol = root(residual, x0, args=(q_prev_val, u_prev_val), tol=1e-8)  # Added tolerance

        if sol.success:
            q_next_val, u_next_val = sol.x
        else:
            # Try with a different solver or more iterations if simple 'root' fails
            # For bounce, this might be sensitive.
            # print(f"Root-finding failed with default: {sol.message}. Trying 'lm'.")
            # sol = root(residual, x0, args=(q_prev_val, u_prev_val), method='lm', tol=1e-8)
            # if sol.success:
            #    q_next_val, u_next_val = sol.x
            # else:
            print(
                f"Root-finding failed for Backward Euler: {sol.message} at t={history['t'][-1]:.2f}, q={q_prev_val:.2f}, u={u_prev_val:.2f}")
            # As a fallback, append previous state or handle error
            q_next_val, u_next_val = q_prev_val, u_prev_val  # Fallback to prevent crash, plot will show issue
            # Or raise RuntimeError("Root-finding failed: " + sol.message)

        history["q"].append(q_next_val)
        history["u"].append(u_next_val)
        history["t"].append(history["t"][-1] + dt)

        q_prev_val, u_prev_val = q_next_val, u_next_val

    return history


def rk4(q0: torch.Tensor, u0: torch.Tensor, dt: float, num_steps: int):
    history = {"t": [0.0],
               "q": [q0.item()],
               "u": [u0.item()]}

    if EXPERIMENT == "pendulum":
        accel = accel_pendulum
    elif EXPERIMENT == "bounce":
        accel = accel_bounce
    else:
        raise ValueError(f"Unknown EXPERIMENT: {EXPERIMENT}")

    q = q0.clone()
    u = u0.clone()
    for _ in range(num_steps):
        k1_q = dt * u
        k1_u = dt * accel(q)

        k2_q = dt * (u + 0.5 * k1_u)
        k2_u = dt * accel(q + 0.5 * k1_q)

        k3_q = dt * (u + 0.5 * k2_u)
        k3_u = dt * accel(q + 0.5 * k2_q)

        k4_q = dt * (u + k3_u)
        k4_u = dt * accel(q + k3_q)

        q_next = q + (k1_q + 2 * k2_q + 2 * k3_q + k4_q) / 6
        u_next = u + (k1_u + 2 * k2_u + 2 * k3_u + k4_u) / 6

        history["q"].append(q_next.item())
        history["u"].append(u_next.item())
        history["t"].append(history["t"][-1] + dt)

        q = q_next
        u = u_next

    return history


def main():
    # Simulation parameters
    # q0_pendulum = 0.2  # initial angle (rad)
    q0_bounce = 2.0  # height of the bounce

    # Set initial conditions based on experiment
    if EXPERIMENT == "pendulum":
        q0_val = 0.2
        y_axis_label = "Angle (rad)"
        plot_title_suffix = "Pendulum"
    elif EXPERIMENT == "bounce":
        q0_val = 2.0
        y_axis_label = "Height (m)"
        plot_title_suffix = "Bounce"
    else:
        raise ValueError(f"Unknown EXPERIMENT: {EXPERIMENT}")

    u0_val = 0.0
    dt = 0.01
    sim_time = 4.0
    steps = int(sim_time / dt)

    q0_tensor = torch.tensor([q0_val])
    u0_tensor = torch.tensor([u0_val])

    # Run the simulation for all integrators
    print("Running Forward Euler...")
    forward_history = forward_euler(q0_tensor, u0_tensor, dt, steps)
    print("Running Symplectic Euler...")
    symplectic_history = symplectic_euler(q0_tensor, u0_tensor, dt, steps)
    print("Running Symplectic Euler (Position)...")
    symplectic_pos_history = symplectic_euler_position(q0_tensor, u0_tensor, dt, steps)
    print("Running Backward Euler...")
    backward_history = backward_euler(q0_tensor, u0_tensor, dt, steps)
    print("Running RK4...")
    rk4_history = rk4(q0_tensor, u0_tensor, dt, steps)
    print("Simulations complete.")

    # Extract results
    forward_q = np.array(forward_history["q"])
    forward_t = np.array(forward_history["t"])

    symplectic_q = np.array(symplectic_history["q"])
    symplectic_t = np.array(symplectic_history["t"])

    symplectic_pos_q = np.array(symplectic_pos_history["q"])
    symplectic_pos_t = np.array(symplectic_pos_history["t"])

    backward_q = np.array(backward_history["q"])
    backward_t = np.array(backward_history["t"])

    rk4_q = np.array(rk4_history["q"])
    rk4_t = np.array(rk4_history["t"])

    # Assert time arrays are consistent
    assert np.allclose(forward_t, symplectic_t), "Time mismatch: FE vs SE"
    assert np.allclose(symplectic_t, symplectic_pos_t), "Time mismatch: SE vs SEP"
    assert np.allclose(symplectic_pos_t, backward_t), "Time mismatch: SEP vs BE"
    assert np.allclose(backward_t, rk4_t), "Time mismatch: BE vs RK4"
    time = forward_t  # Use one of the time arrays

    # Prepare data for create_scientific_subplot_plot
    plot_data_for_subplot_plot = {
        f'Position vs. Time ({plot_title_suffix})': {
            # 'Forward Euler': {
            #     'data': forward_q,
            #     'label': 'Forward Euler',
            #     # 'color': '#1F77B4', # Blue
            #     'color': '#FF7F0E', # Orange
            #     'linewidth': 2,
            #     'alpha': 1.0
            # },
            'Symplectic Euler': {
                'data': symplectic_q,
                'label': 'Symplectic Euler',
                # 'color': '#FF7F0E', # Orange
                'color': '#1F77B4', # Blue
                'linewidth': 2
            },
            # 'Symplectic Euler (Pos)': {
            #     'data': symplectic_pos_q,
            #     'label': 'Symplectic Euler (Position)',
            #     'color': 'green',
            #     'linewidth': 1.5,
            #     'linestyle': '--'
            # },
            # 'Backward Euler': {
            #     'data': backward_q,
            #     'label': 'Backward Euler',
            #     'color': '#2CA02C', # Green
            #     'linewidth': 2,
            #     'alpha': 1.0
            # },
            'RK4': {
                'data': rk4_q,
                'label': 'RK4',
                # 'color': '#6B42F0', # Purple
                'color': '#D62728',
                'linestyle': '--',
                'linewidth': 2
            }
        }
    }

    config_for_subplot_plot = {
        'plot_arrangement': 'horizontal',
        'figsize': (8, 5),  # Adjusted for potentially more legend items
        'suptitle': f'',
        # 'suptitle': f'Integrator Comparison ({plot_title_suffix} Experiment)',
        'x_label': 'Time [s]',
        'y_labels': {f'Position vs. Time ({plot_title_suffix})': y_axis_label},
        'plot_titles': {f'Position vs. Time ({plot_title_suffix})': f''},
        'legend': {'show': True, 'location': 'best', 'fontsize': 9},  # Smaller legend fontsize
        'grid': {'show': True},
        'font_sizes': {'axis_label': 11, 'tick_label': 12, 'title': 12, 'suptitle': 14}
    }

    fig, ax = create_scientific_subplot_plot(
        time_data=time,
        plot_data=plot_data_for_subplot_plot,
        config=config_for_subplot_plot,
        save_path=f"integrators_comparison_{EXPERIMENT}.png"  # Save path for the plot
    )

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    # You can switch the experiment here if needed
    # EXPERIMENT = "pendulum"
    # EXPERIMENT = "bounce"
    main()