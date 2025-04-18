import torch
import matplotlib.pyplot as plt
from xitorch.integrate import solve_ivp


# Dynamics function for a pendulum: dy/dt = [y[1], -g/l * sin(y[0])]
def y_grad(t: torch.Tensor, y: torch.Tensor, *params) -> torch.Tensor:
    g, l, dt = params
    dydt = torch.zeros_like(y)
    dydt[0] = y[1]  # dtheta/dt = omega
    dydt[1] = -g / l * torch.sin(y[0])  # domega/dt = -g/l * sin(theta)
    return dydt

def f(t: torch.Tensor, y: torch.Tensor, *params) -> torch.Tensor:
    g, l, dt = params

    y_next = y + dt * y_grad(t, y, *params)

    # Newton iterations to solve y_next = y_curr + dt * f(t_next, y_next)
    for _ in range(100):
        dy_next = y_grad(t, y_next, *params)
        residual = y_next - (y + dt * dy_next)

        if torch.all(torch.abs(residual) < 1e-9):
            break

        # Jacobian: I - dt * df/dy_next
        dres_dy_next = compute_jacobian(t, dy_next, *params)
        J = torch.eye(2, dtype=y_next.dtype, device=y_next.device) - dt * dres_dy_next
        delta = torch.linalg.solve(J, -residual)
        y_next = y_next + delta

    return (y_next - y) / dt

# Analytical Jacobian of f with respect to y
def compute_jacobian(t: torch.Tensor, y: torch.Tensor, *params) -> torch.Tensor:
    g, l, _ = params
    J = torch.zeros((2, 2), dtype=y.dtype, device=y.device)
    J[0, 1] = 1.0  # df0/dy1
    J[1, 0] = -g / l * torch.cos(y[0])  # df1/dy0
    return J


# @torch.no_grad()
def backward_euler_integrator(fcn, ts, y0, params, **kwargs):
    _, _, dt = params
    yt = torch.empty((len(ts), *y0.shape), dtype=y0.dtype, device=y0.device)
    yt[0] = y0

    for i in range(len(ts) - 1):
        t_curr, y_curr = ts[i], yt[i]
        t_next = ts[i + 1]

        yt[i + 1] = y_curr + dt * fcn(t_next, y_curr, *params)

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
        torch.tensor(dt, requires_grad=True) # time step
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


def simulate(g, l, dt, y0, ts):
    # Wrap parameters
    params = [
        torch.tensor(g, requires_grad=True),  # g
        torch.tensor(l, requires_grad=True),  # l
        torch.tensor(dt, requires_grad=True)  # time step
    ]

    # Convert y0 to tensor if it's not already
    if not isinstance(y0, torch.Tensor):
        y0 = torch.tensor(y0, dtype=torch.float64, requires_grad=True)

    yt = solve_ivp(
        f,
        ts,
        y0,
        method=backward_euler_integrator,
        params=params
    )

    return yt, params


def numerical_gradient_test():
    # Simulation parameters
    q0 = 2.0  # initial position/angle
    u0 = 0.0  # initial velocity
    dt = 0.005  # time step (reduced to speed up the test)
    sim_time = 0.3  # reduced simulation time for testing
    steps = int(sim_time / dt)

    # Base parameters
    g_base = 9.81
    l_base = 1.0
    y0 = torch.tensor([q0, u0], dtype=torch.float64)
    ts = torch.linspace(0, sim_time, steps + 1)

    # Small perturbation for finite difference
    epsilon = 1e-4

    print("Testing gradient with respect to g (gravity):")

    # Run the simulation with base parameters for g gradient
    g_tensor = torch.tensor(g_base, dtype=torch.float64, requires_grad=True)
    l_tensor = torch.tensor(l_base, dtype=torch.float64, requires_grad=True)
    dt_tensor = torch.tensor(dt, dtype=torch.float64, requires_grad=True)

    params = [g_tensor, l_tensor, dt_tensor]

    yt_base = solve_ivp(f, ts, y0, method=backward_euler_integrator, params=params)

    # Get analytical gradient using autograd
    analytical_grad_g = torch.autograd.grad(yt_base[-1, 0], g_tensor, retain_graph=True)[0]

    # Compute numerical gradient for g using finite differences
    g_plus = torch.tensor(g_base + epsilon, dtype=torch.float64)
    g_minus = torch.tensor(g_base - epsilon, dtype=torch.float64)

    params_plus = [g_plus, l_tensor, dt_tensor]
    params_minus = [g_minus, l_tensor, dt_tensor]

    yt_plus = solve_ivp(f, ts, y0, method=backward_euler_integrator, params=params_plus)
    yt_minus = solve_ivp(f, ts, y0, method=backward_euler_integrator, params=params_minus)

    numerical_grad_g = (yt_plus[-1, 0].detach() - yt_minus[-1, 0].detach()) / (2 * epsilon)

    print(f"Analytical gradient for g: {analytical_grad_g.item()}")
    print(f"Numerical gradient for g: {numerical_grad_g.item()}")
    print(f"Absolute difference: {abs(analytical_grad_g.item() - numerical_grad_g.item())}")
    print(
        f"Relative difference: {abs(analytical_grad_g.item() - numerical_grad_g.item()) / max(abs(analytical_grad_g.item()), abs(numerical_grad_g.item()))}")

    print("\nTesting gradient with respect to l (pendulum length):")

    # Run the simulation with base parameters for l gradient
    g_tensor = torch.tensor(g_base, dtype=torch.float64, requires_grad=True)
    l_tensor = torch.tensor(l_base, dtype=torch.float64, requires_grad=True)
    dt_tensor = torch.tensor(dt, dtype=torch.float64, requires_grad=True)

    params = [g_tensor, l_tensor, dt_tensor]

    yt_base = solve_ivp(f, ts, y0, method=backward_euler_integrator, params=params)

    # Get analytical gradient using autograd
    analytical_grad_l = torch.autograd.grad(yt_base[-1, 0], l_tensor, retain_graph=True)[0]

    # Compute numerical gradient for l using finite differences
    l_plus = torch.tensor(l_base + epsilon, dtype=torch.float64)
    l_minus = torch.tensor(l_base - epsilon, dtype=torch.float64)

    params_plus = [g_tensor, l_plus, dt_tensor]
    params_minus = [g_tensor, l_minus, dt_tensor]

    yt_plus = solve_ivp(f, ts, y0, method=backward_euler_integrator, params=params_plus)
    yt_minus = solve_ivp(f, ts, y0, method=backward_euler_integrator, params=params_minus)

    numerical_grad_l = (yt_plus[-1, 0].detach() - yt_minus[-1, 0].detach()) / (2 * epsilon)

    print(f"Analytical gradient for l: {analytical_grad_l.item()}")
    print(f"Numerical gradient for l: {numerical_grad_l.item()}")
    print(f"Absolute difference: {abs(analytical_grad_l.item() - numerical_grad_l.item())}")
    print(
        f"Relative difference: {abs(analytical_grad_l.item() - numerical_grad_l.item()) / max(abs(analytical_grad_l.item()), abs(numerical_grad_l.item()))}")

    # Test initial position gradient
    print("\nTesting gradient with respect to initial angle:")

    # Run the simulation with base parameters for initial state gradient
    g_tensor = torch.tensor(g_base, dtype=torch.float64, requires_grad=True)
    l_tensor = torch.tensor(l_base, dtype=torch.float64, requires_grad=True)
    dt_tensor = torch.tensor(dt, dtype=torch.float64, requires_grad=True)
    y0_grad = torch.tensor([q0, u0], dtype=torch.float64, requires_grad=True)

    params = [g_tensor, l_tensor, dt_tensor]

    yt_base = solve_ivp(f, ts, y0_grad, method=backward_euler_integrator, params=params)

    # Get analytical gradient using autograd
    analytical_grad_q0 = torch.autograd.grad(yt_base[-1, 0], y0_grad, retain_graph=True)[0][0]

    # Compute numerical gradient for initial angle using finite differences
    y0_plus = torch.tensor([q0 + epsilon, u0], dtype=torch.float64, requires_grad=True)
    y0_minus = torch.tensor([q0 - epsilon, u0], dtype=torch.float64, requires_grad=True)

    yt_plus = solve_ivp(f, ts, y0_plus, method=backward_euler_integrator, params=params)
    yt_minus = solve_ivp(f, ts, y0_minus, method=backward_euler_integrator, params=params)

    numerical_grad_q0 = (yt_plus[-1, 0].detach() - yt_minus[-1, 0].detach()) / (2 * epsilon)

    print(f"Analytical gradient for initial angle: {analytical_grad_q0.item()}")
    print(f"Numerical gradient for initial angle: {numerical_grad_q0.item()}")
    print(f"Absolute difference: {abs(analytical_grad_q0.item() - numerical_grad_q0.item())}")
    print(
        f"Relative difference: {abs(analytical_grad_q0.item() - numerical_grad_q0.item()) / max(abs(analytical_grad_q0.item()), abs(numerical_grad_q0.item()))}")


if __name__ == "__main__":
    main()
    numerical_gradient_test()
