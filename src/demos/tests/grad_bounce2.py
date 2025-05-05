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

def backward_euler_integrator(fcn, ts, y0, params, restitution=0.0):
    g = params[0]
    e = restitution
    n_steps = len(ts) - 1
    yt_list = [y0]  # y0 is [x, y, vx, vy]

    for i in range(n_steps):
        t_curr, y_curr = ts[i], yt_list[i]
        t_next = ts[i + 1]
        dt = t_next - t_curr

        # Residual function with contact
        def residual(z_next, y_curr, dt, g, collision: bool):
            vx_next, vy_next, lambda_next = z_next
            # Velocity updates
            res1 = vx_next - y_curr[2]  # No force in x-direction
            res2 = vy_next - y_curr[3] + dt * g - lambda_next
            # Complementarity with restitution: λ ≥ 0, y ≥ 0, λ * (y - e * v_{y,prev} * dt) = 0
            if collision:
                with torch.no_grad():
                    b_err = (0.01 / dt) * y_curr[1]
                b_rest = e * y_curr[3]
                res3 = fb(lambda_next, vy_next + b_err + b_rest)
            else:
                res3 = -lambda_next
            return torch.stack([res1, res2, res3])

        # Initial guess: forward Euler without contact
        x_guess = y_curr[0] + dt * y_curr[2]
        y_guess = y_curr[1] + dt * y_curr[3]
        vx_guess = y_curr[2]
        vy_guess = y_curr[3] - dt * g
        z_guess = torch.tensor([vx_guess, vy_guess, 0.0], dtype=torch.float64)

        # Check for collision
        collision = False
        if y_curr[1] < 0.0:
            collision = True

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
        yt_list.append(torch.stack([x_next, y_next, vx_next, vy_next]))

    return torch.stack(yt_list, dim=0)

def compute_loss(py_init, target_height, ts, params, restitution=0.0):
    """Compute loss based on final y-position (height) only."""
    py_init = torch.tensor(py_init, dtype=torch.float64, requires_grad=True) if not isinstance(py_init, torch.Tensor) else py_init
    y0 = torch.stack([
        torch.tensor(0.0, dtype=torch.float64),
        py_init,
        torch.tensor(5.0, dtype=torch.float64),
        torch.tensor(-5.0, dtype=torch.float64)
    ])
    yt = backward_euler_integrator(f, ts, y0, params=params, restitution=restitution)
    final_height = yt[-1, 1]  # y-position only
    loss = (final_height - target_height) ** 2
    return loss, yt


def main():
    # Simulation parameters
    px0 = 0.0           # Initial x-position
    py_target = 2.1     # Initial y-position
    vx0 = 5.0           # Initial x-velocity
    vy0 = -5.0          # Target initial y-velocity
    dt = 0.05           # Time step
    sim_time = 0.8      # Simulation time
    steps = int(sim_time / dt)
    restitution = 1.0  # Coefficient of restitution

    # Parameters
    params = [torch.tensor(0.0, dtype=torch.float64, requires_grad=True)]  # g

    # Time steps
    ts = torch.linspace(0, sim_time, steps + 1)

    # Compute target height with vx=5.0
    y0_target = torch.tensor([px0, py_target, vx0, vy0], dtype=torch.float64, requires_grad=True)
    yt_target = backward_euler_integrator(f, ts, y0_target, params=params, restitution=restitution)
    target_height = yt_target[-1, 1]  # Only the final y-position
    print(f"Target height with y0 = {py_target} m: {target_height.detach().numpy():.6f} m")

    # Plot target trajectory
    t_np = ts.detach().numpy()
    pos_np = yt_target[:, :2].detach().numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(pos_np[:, 0], pos_np[:, 1], 'b-', label=f'Trajectory (y0={py_target:.2f})')
    plt.scatter(pos_np[0, 0], pos_np[0, 1], c='g', s=100, label='Start')
    plt.scatter(pos_np[-1, 0], pos_np[-1, 1], c='r', s=100, label='End')
    plt.axhline(y=target_height.detach().numpy(), color='k', linestyle='--', label=f'Target Height ({target_height.detach().numpy():.2f} m)')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Target Trajectory')
    plt.legend()
    plt.grid(True)
    plt.savefig('target_trajectory.png')
    plt.close()

    # Test loss and gradient over vx range 0 to 10 m/s
    py_range = np.linspace(0.1, 4.1, 100)
    loss_values = []
    grad_values = []
    for py in py_range:
        py_tensor = torch.tensor(py, dtype=torch.float64, requires_grad=True)
        loss, _ = compute_loss(py_tensor, target_height, ts, params=params, restitution=restitution)
        grad_py = torch.autograd.grad(loss, py_tensor)[0]
        print(f"y: {py:.2f}, Loss: {loss.item():.6f}, Gradient: {grad_py.item():.6f}")
        loss_values.append(loss.item())
        grad_values.append(grad_py.item())

    # Plot loss and gradient curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(py_range, loss_values, 'b-')
    ax1.set_xlabel('Initial Y-Position (m/s)')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Initial Y-Position')
    ax1.grid(True)
    ax2.plot(py_range, grad_values, 'r-')
    ax2.set_xlabel('Initial Y-Position (m/s)')
    ax2.set_ylabel('Gradient')
    ax2.set_title('Gradient vs Initial Y-Position')
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig('loss_gradient.png')
    plt.close()

if __name__ == "__main__":
    main()