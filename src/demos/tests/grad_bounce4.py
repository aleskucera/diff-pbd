import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from xitorch.optimize import rootfinder

torch.autograd.set_detect_anomaly(True)

# Fischer-Burmeister function for complementarity
def fb(a: torch.Tensor, b: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    return a + b - torch.sqrt(a**2 + b**2 + epsilon)

# Dynamics function (free-fall under gravity, no contact)
def f(t: torch.Tensor, y: torch.Tensor, *params) -> torch.Tensor:
    g = params[0]
    dx = y[2]         # dx/dt = vx
    dy = y[3]         # dy/dt = vy
    dvx = 0.0         # dvx/dt = 0
    dvy = -g          # dvy/dt = -g
    return torch.stack([dx, dy, dvx, dvy], dim=0)

def newton(fcn, x0, params, **config):
    tol = config["tol"]
    maxiter = config["maxiter"]
    x = x0
    for _ in range(maxiter):
        fx = fcn(x, *params)
        J = torch.autograd.functional.jacobian(lambda y: fcn(y, *params), x)
        delta = torch.linalg.solve(J, -fx)
        x = x + delta
    return x

def backward_euler_step(y_curr, dt, g, restitution, collision):
    """Perform a single backward Euler step with contact handling."""
    e = restitution

    # Residual function with contact
    def residual(z_next, y_curr, dt, g, collision: bool):
        vx_next, vy_next, lambda_next = z_next
        res1 = vx_next - y_curr[2]  # No force in x-direction
        res2 = vy_next - y_curr[3] + dt * g - lambda_next
        if collision:
            b_err = (0.0 / dt) * y_curr[1]
            b_rest = e * y_curr[3]
            res3 = fb(lambda_next, vy_next + b_err + b_rest)
        else:
            res3 = -lambda_next
        return torch.stack([res1, res2, res3])

    # Initial guess: forward Euler without contact
    vx_guess = y_curr[2]
    vy_guess = y_curr[3] - dt * g
    z_guess = torch.tensor([vx_guess, vy_guess, 0.0], dtype=torch.float64)

    # Solve for next state
    z_next = rootfinder(
        residual,
        z_guess,
        method=newton,
        params=(y_curr, dt, g, collision),
        tol=1e-9,
        maxiter=10
    )

    # Extract state
    vx_next, vy_next = z_next[:2]
    x_next = y_curr[0] + dt * vx_next
    y_next = y_curr[1] + dt * vy_next
    y_next_state = torch.stack([x_next, y_next, vx_next, vy_next])
    return y_next_state, z_next[2]  # Return state and contact force

def adaptive_backward_euler_integrator(fcn, t_span, y0, params, restitution=0.0, tol=1e-3, dt_init=0.05, dt_min=1e-6, dt_max=0.1):
    """Backward Euler integrator with adaptive time stepping based on penetration depth."""
    g = params[0]
    t_start, t_end = t_span
    t_curr = t_start
    y_curr = y0
    dt = dt_init
    e = restitution

    # Lists to store results
    times = [t_curr]
    states = [y_curr]
    contact_forces = []

    while t_curr < t_end:
        # Check for collision based on current position
        collision = y_curr[1] < 0.0

        # Trial step
        y_next, lambda_next = backward_euler_step(y_curr, dt, g, e, collision)

        # Compute penetration error
        penetration_error = torch.abs(y_next[1]) if y_next[1] < 0 else torch.tensor(0.0, dtype=torch.float64)

        # Accept or reject step based on error
        if penetration_error <= tol or y_next[1] >= 0:
            # Accept step
            t_curr = t_curr + dt
            y_curr = y_next
            times.append(t_curr)
            states.append(y_curr)
            contact_forces.append(lambda_next)

            # Adjust time step for next iteration
            if penetration_error > 0:
                # Scale time step based on error (proportional control)
                safety_factor = 0.9
                dt_new = dt * safety_factor * torch.sqrt(tol / penetration_error)
                dt = torch.clamp(dt_new, dt_min, dt_max)
            else:
                # Increase time step if no penetration (e.g., in free flight)
                dt = torch.min(torch.tensor(dt * 1.2), torch.tensor(dt_max))
        else:
            # Reject step and reduce time step
            dt = torch.max(torch.tensor(dt * 0.5), torch.tensor(dt_min))

        # Ensure we don't overshoot the end time
        if t_curr + dt > t_end:
            dt = t_end - t_curr

    return torch.tensor(times, dtype=torch.float64), torch.stack(states, dim=0), torch.stack(contact_forces, dim=0)

def compute_loss(py_init, target_height, t_span, params):
    """Compute loss based on final y-position (height) only."""
    py_init = torch.tensor(py_init, dtype=torch.float64, requires_grad=True) if not isinstance(py_init, torch.Tensor) else py_init
    y0 = torch.stack([
        torch.tensor(0.0, dtype=torch.float64),
        py_init,
        torch.tensor(5.0, dtype=torch.float64),
        torch.tensor(-5.0, dtype=torch.float64)
    ])
    ts, yt, _ = adaptive_backward_euler_integrator(f, t_span, y0, params=params)
    final_height = yt[-1, 1]  # y-position only
    loss = (final_height - target_height) ** 2
    return loss, yt, ts

def main():
    # Simulation parameters
    px0 = 0.0           # Initial x-position
    py_target = 2.1     # Initial y-position
    vx0 = 5.0           # Initial x-velocity
    vy0 = -5.0          # Target initial y-velocity
    sim_time = 0.5      # Simulation time
    restitution = 0.3   # Coefficient of restitution
    tol = 1e-3          # Penetration tolerance
    dt_init = 0.05      # Initial time step
    dt_min = 1e-6       # Minimum time step
    dt_max = 0.1        # Maximum time step

    # Parameters
    params = [torch.tensor(0.0, dtype=torch.float64)]  # g

    # Time span
    t_span = (0.0, sim_time)

    # Initial state (which will be the target for the optimization)
    y0_target = torch.tensor([px0, py_target, vx0, vy0], dtype=torch.float64)
    ts_target, yt_target, lambda_target = adaptive_backward_euler_integrator(
        f, t_span, y0_target, params=params, restitution=restitution, tol=tol, dt_init=dt_init, dt_min=dt_min, dt_max=dt_max
    )
    target_height = yt_target[-1, 1]
    print(f"Target height with y0 = {py_target} m: {target_height.detach().numpy():.6f} m")

    # Plot target trajectory
    t_np = ts_target.detach().numpy()
    pos_np = yt_target[:, :2].detach().numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(pos_np[:, 0], pos_np[:, 1], 'b-', label=f'Trajectory (y0={py_target:.2f})')
    plt.scatter(pos_np[0, 0], pos_np[0, 1], c='g', s=100, label='Start')
    plt.scatter(pos_np[-1, 0], pos_np[-1, 1], c='r', s=100, label='End')
    plt.axhline(y=target_height.detach().numpy(), color='k', linestyle='--', label=f'Target Height ({target_height.detach().numpy():.2f} m)')
    plt.axhline(y=0.0, color='gray', linestyle='-', label='Ground')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Target Trajectory with Adaptive Time Stepping')
    plt.legend()
    plt.grid(True)
    plt.savefig('target_trajectory_adaptive.png')
    plt.close()

    # Plot time step sizes
    dt_np = np.diff(t_np)
    plt.figure(figsize=(10, 6))
    plt.plot(t_np[:-1], dt_np, 'b-', label='Time Step Size')
    plt.xlabel('Time (s)')
    plt.ylabel('Time Step (s)')
    plt.title('Time Step Size vs Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('time_steps.png')
    plt.close()

    # Test loss and gradient over y0 range 1.8 to 2.4 m
    py_range = np.linspace(1.8, 2.4, 50)
    loss_values = []
    grad_values = []
    for py in py_range:
        py_tensor = torch.tensor(py, dtype=torch.float64, requires_grad=True)
        loss, _, _ = compute_loss(py_tensor, target_height, t_span, params)
        grad_py = torch.autograd.grad(loss, py_tensor)[0]
        print(f"y0: {py:.2f}, Loss: {loss.item():.6f}, Gradient: {grad_py.item():.6f}")
        loss_values.append(loss.item())
        grad_values.append(grad_py.item())

    # Plot loss and gradient curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(py_range, loss_values, 'b-')
    ax1.set_xlabel('Initial Y-Position (m)')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Initial Y-Position')
    ax1.grid(True)
    ax2.plot(py_range, grad_values, 'r-')
    ax2.set_xlabel('Initial Y-Position (m)')
    ax2.set_ylabel('Gradient')
    ax2.set_title('Gradient vs Initial Y-Position')
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig('loss_gradient_adaptive.png')
    plt.close()

if __name__ == "__main__":
    main()