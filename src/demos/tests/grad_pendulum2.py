import torch
import matplotlib.pyplot as plt
from xitorch.integrate import solve_ivp


# Dynamics function for a pendulum: dy/dt = [y[1], -g/l * sin(y[0])]
def f(t: torch.Tensor, y: torch.Tensor, *params) -> torch.Tensor:
    g, l, dt = params
    dydt = torch.zeros_like(y)
    dydt[0] = y[1]  # dtheta/dt = omega
    dydt[1] = -g / l * torch.sin(y[0])  # domega/dt = -g/l * sin(theta)
    return dydt


# Analytical Jacobian of f with respect to y
def compute_jacobian(t: torch.Tensor, y: torch.Tensor, *params) -> torch.Tensor:
    g, l, dt = params
    J = torch.zeros((2, 2), dtype=y.dtype, device=y.device)
    J[0, 1] = 1.0  # df0/dy1
    J[1, 0] = -g / l * torch.cos(y[0])  # df1/dy0
    return J


# Custom backward Euler integrator
def backward_euler_integrator(fcn, ts, y0, params, **kwargs):
    g, l, dt = params
    yt = torch.empty((len(ts), *y0.shape), dtype=y0.dtype, device=y0.device)
    yt[0] = y0

    for i in range(len(ts) - 1):
        t_curr, y_curr = ts[i], yt[i]
        t_next = ts[i + 1]

        # Initial guess using forward Euler
        y_next = y_curr + dt * fcn(t_curr, y_curr, *params)

        # Newton iterations to solve y_next = y_curr + dt * f(t_next, y_next)
        for _ in range(100):
            dy_next = fcn(t_next, y_next, *params)
            residual = y_next - (y_curr + dt * dy_next)

            if torch.all(torch.abs(residual) < 1e-9):
                break

            # Jacobian: I - dt * df/dy_next
            dres_dy_next = compute_jacobian(t_next, dy_next, *params)
            J = torch.eye(2, dtype=y_next.dtype, device=y_next.device) - dt * dres_dy_next
            delta = torch.linalg.solve(J, -residual)
            y_next = y_next + delta

        yt[i + 1] = y_next

    return yt

def main():
    # Simulation parameters
    q0 = 2.0  # initial position/angle
    u0 = 0.0  # initial velocity
    dt = 0.001  # time step (s)
    sim_time = 5.0  # total simulation time (s)
    steps = int(sim_time / dt)  # number of time steps

    # Parameters
    params = [
        torch.tensor(9.81, requires_grad=True), # g
        torch.tensor(1.0, requires_grad=True), # l
        torch.tensor(dt, requires_grad=True) # dt
    ]

    # Create time steps tensor
    ts = torch.linspace(0, sim_time, steps + 1)
    y0 = torch.tensor([q0, u0], dtype=torch.float64, requires_grad=True)  # initial state

    yt = solve_ivp(
        f,
        ts,
        y0,
        method=backward_euler_integrator,
        params=params
    )

    # first order grad
    grad_g, = torch.autograd.grad(yt[-1, 0], params[0], create_graph=True)
    print("Gradient with respect to g:", grad_g)

    # Convert to numpy for plotting
    time = ts.detach().numpy()
    yt = yt[:, 0].detach().numpy()

    # Plot positions over time
    plt.figure(figsize=(10, 6))
    plt.plot(time, yt, label="Backward Euler", linestyle="-", alpha=0.8, color="red", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [rad]")
    plt.title(f"Position vs. Time")
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    main()