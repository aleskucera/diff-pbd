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
