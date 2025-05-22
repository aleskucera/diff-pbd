import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from xitorch.optimize import rootfinder

# torch.autograd.set_detect_anomaly(True)

# Fischer-Burmeister function for complementarity
def fb(a: torch.Tensor, b: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    return a + b - torch.sqrt(a**2 + b**2 + epsilon)


def fb_stable(a: torch.Tensor, b: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Fischer-Burmeister function with explicit gradient handling."""

    # Define custom autograd function for the FB function
    class StableFB(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b, epsilon):
            ctx.save_for_backward(a, b)
            ctx.epsilon = epsilon

            # Safe computation for forward pass
            squared_sum = a ** 2 + b ** 2 + epsilon
            norm = torch.sqrt(squared_sum)
            result = a + b - norm
            return result

        @staticmethod
        def backward(ctx, grad_output):
            a, b = ctx.saved_tensors
            epsilon = ctx.epsilon

            # Manually compute gradients
            with torch.no_grad():
                squared_sum = a ** 2 + b ** 2 + epsilon
                norm = torch.sqrt(squared_sum)

                # Safe computation of partial derivatives
                da = 1.0 - a / torch.clamp(norm, min=epsilon)
                db = 1.0 - b / torch.clamp(norm, min=epsilon)

                # Handle extreme cases
                extreme_case = squared_sum < epsilon
                da = torch.where(extreme_case, torch.ones_like(da), da)
                db = torch.where(extreme_case, torch.ones_like(db), db)

            # Apply chain rule
            grad_a = grad_output * da
            grad_b = grad_output * db
            grad_epsilon = None  # Not needed

            return grad_a, grad_b, grad_epsilon

    # Use our custom function
    return StableFB.apply(a, b, epsilon)


# Dynamics function (free-fall under gravity, no contact)
def forward_euler(t: torch.Tensor, y: torch.Tensor, *params) -> torch.Tensor:
    g = params[0]
    dx = y[2]         # dx/dt = vx
    dy = y[3]         # dy/dt = vy
    dvx = torch.tensor(0.0, dtype=torch.float64, device=y.device)  # dvx/dt = 0
    dvy = -g          # dvy/dt = -g (g is already a tensor)
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
            with torch.no_grad():
                b_err = (0.01 / dt) * y_curr[1]
            b_rest = e * y_curr[3]
            res3 = fb_stable(lambda_next, vy_next + b_err + b_rest)
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


def free_fall_substeps(t_curr, y_curr, dt, num_substeps, g):
    """Compute free-fall states at substep times without contact constraints."""
    with torch.no_grad():
        sub_times = torch.linspace(t_curr, t_curr + dt, num_substeps, dtype=torch.float64, device=y_curr.device)
        sub_states = torch.zeros((num_substeps, 4), dtype=torch.float64, device=y_curr.device)
        sub_states[0] = y_curr
        sub_dt = dt / (num_substeps - 1) if num_substeps > 1 else dt

        for i in range(1, num_substeps):
            # Forward Euler step for free fall (no contact forces)
            y_prev = sub_states[i - 1]
            dydt = forward_euler(sub_times[i - 1], y_prev, g)
            y_next = y_prev + sub_dt * dydt
            sub_states[i] = y_next

    return sub_times, sub_states


def adaptive_backward_euler_integrator(fcn, t_span, y0, params, restitution=0.0, tol=1e-3, dt_init=0.01, num_substeps=10):
    g = params[0]
    t_start, t_end = t_span
    t_curr = t_start
    y_curr = y0
    dt = dt_init
    e = restitution

    # Lists to store results
    times = [t_curr]
    states = [y_curr]

    while t_curr < t_end:
        with torch.no_grad():
            sub_times, sub_states = free_fall_substeps(t_curr, y_curr, dt, num_substeps, g)

            # Check for penetration (y < -tol)
            y_positions = sub_states[:, 1]
            penetration_mask = y_positions < -0.001

            # Compute the last non-penetration time and first penetration time
            last_non_penetration_idx = torch.where(~penetration_mask)[0][-1] if not penetration_mask.all() else None
            first_penetration_idx = torch.where(penetration_mask)[0][0] if penetration_mask.any() else None

        if last_non_penetration_idx is not None:
            # Do the forward Euler step right before collision
            dt_forward = sub_times[last_non_penetration_idx] - t_curr

            if t_curr + dt_forward > t_end:
                dt = t_end - t_curr
                if dt == 0:
                    break

            dydt = forward_euler(t_curr, y_curr, g)
            y_curr = y_curr + dt_forward * dydt
            t_curr = t_curr + dt_forward
            states.append(y_curr)
            times.append(t_curr)

        if first_penetration_idx is not None:
            # Collision detected, implicitly compute the step
            collision = True
            dt_backward = sub_times[first_penetration_idx] - t_curr

            if t_curr + dt_backward > t_end:
                dt = t_end - t_curr
                if dt == 0:
                    break

            y_next, lambda_next = backward_euler_step(y_curr, dt_backward, g, e, collision)

            y_curr = y_next
            t_curr = t_curr + dt_backward
            states.append(y_curr)
            times.append(t_curr)

    return torch.tensor(times, dtype=torch.float64), torch.stack(states, dim=0)

def compute_loss(py_init, target_height, t_span, params, restitution=0.0):
    """Compute loss based on final y-position (height) only."""
    py_init = torch.tensor(py_init, dtype=torch.float64, requires_grad=True) if not isinstance(py_init, torch.Tensor) else py_init
    y0 = torch.stack([
        torch.tensor(0.0, dtype=torch.float64),
        py_init,
        torch.tensor(5.0, dtype=torch.float64),
        torch.tensor(-5.0, dtype=torch.float64)
    ])
    ts, yt = adaptive_backward_euler_integrator(forward_euler, t_span, y0, params=params, restitution=restitution)
    final_height = yt[-1, 1]  # y-position only
    loss = final_height - target_height
    return loss, yt, ts

def main():
    # Simulation parameters
    px0 = 0.0           # Initial x-position
    py_target = 2.1     # Initial y-position
    vx0 = 5.0           # Initial x-velocity
    vy0 = -5.0          # Target initial y-velocity
    sim_time = 1.0      # Simulation time
    restitution = 1.0   # Coefficient of restitution
    tol = 1e-3          # Penetration tolerance
    dt_init = 0.01      # Initial time step
    num_substeps = 100  # Number of substeps for preview

    # Parameters
    params = [torch.tensor(0.0, dtype=torch.float64, requires_grad=True)]  # g

    # Time span
    t_span = (0.0, sim_time)

    # Compute target height with y0 = 2.1
    y0_target = torch.tensor([px0, py_target, vx0, vy0], dtype=torch.float64, requires_grad=True)

    integration_start_time = time.time()
    ts_target, yt_target = adaptive_backward_euler_integrator(
        forward_euler, t_span, y0_target, params=params, restitution=restitution, tol=tol, dt_init=dt_init, num_substeps=num_substeps
    )
    integration_end_time = time.time()
    print(f"Integration time: {integration_end_time - integration_start_time:.6f}")
    target_height = yt_target[-1, 1]  # Final y-position
    print(f"Target height with y0 = {py_target} m: {target_height.detach().numpy():.6f} m")
    print(f"Number of time steps: {len(ts_target)}")

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
    plt.title('Target Trajectory with Substep-Based Adaptive Time Stepping')
    plt.legend()
    plt.grid(True)
    plt.savefig('target_trajectory_substep.png')
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
    plt.savefig('time_steps_substep.png')
    plt.close()

    # Test loss and gradient over y0 range 1.8 to 2.4 m
    py_range = np.linspace(0.1, 4.1, 100)
    loss_values = []
    grad_values = []
    for py in py_range:
        py_tensor = torch.tensor(py, dtype=torch.float64, requires_grad=True)
        loss_start_time = time.time()
        loss, _, _ = compute_loss(py_tensor, target_height, t_span, params, restitution=restitution)
        loss_end_time = time.time()
        print(f"Loss computation time: {loss_end_time - loss_start_time:.6f}")

        grad_start_time = time.time()
        grad_py = torch.autograd.grad(loss, py_tensor)[0]
        grad_end_time = time.time()
        print(f"Grad computation time: {grad_end_time - grad_start_time:.6f}")
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
    plt.savefig('loss_gradient_substep.png')
    plt.close()

if __name__ == "__main__":
    main()
