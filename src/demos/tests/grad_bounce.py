import time
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from xitorch.optimize import rootfinder

torch.autograd.set_detect_anomaly(True)

# Define the Fischer-Burmeister function for complementarity
def fb(a: torch.Tensor, b: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    return a + b - torch.sqrt(a**2 + b**2 + epsilon)

# Define the dynamics function (free-fall under gravity, without contact)
def f(t: torch.Tensor, y: torch.Tensor, *params) -> torch.Tensor:
    g = params[0]
    dy = y[1]         # dy/dt = v
    dv = -g           # dv/dt = -g
    return torch.stack([dy, dv], dim=0)

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

def backward_euler_integrator(fcn, ts, y0, params, **kwargs):
    g = params[0]  # Gravity as the only parameter
    n_steps = len(ts) - 1

    # Store trajectory as a list
    yt_list = [y0]  # y0 is [y, v]

    for i in range(n_steps):
        t_curr, y_curr = ts[i], yt_list[i]
        t_next = ts[i + 1]
        dt = t_next - t_curr

        # Residual function including contact via complementarity
        def residual(z_next, y_curr, dt, g):
            y_next, v_next, lambda_next = z_next[0], z_next[1], z_next[2]
            res1 = y_next - y_curr[0] - dt * v_next              # Position update
            res2 = v_next - y_curr[1] + dt * g - lambda_next     # Velocity update with impulse
            res3 = fb(lambda_next, y_next)                       # Complementarity: 0 <= lambda ⊥ y >= 0
            return torch.stack([res1, res2, res3])

        # Initial guess: forward Euler without contact, lambda = 0
        y_guess = y_curr[0] + dt * y_curr[1]
        v_guess = y_curr[1] - dt * g
        z_guess = torch.tensor([y_guess, v_guess, 0.0], dtype=torch.float64)

        # Solve for next state [y_next, v_next, lambda_next]
        z_next = rootfinder(
            residual,
            z_guess,
            method=newton,
            params=(y_curr, dt, g),
            tol=1e-9,
            maxiter=10
        )

        # Extract position and velocity, discard lambda
        y_next, v_next = z_next[0], z_next[1]
        yt_list.append(torch.stack([y_next, v_next]))

    # Stack trajectory
    yt = torch.stack(yt_list, dim=0)
    return yt

def main():
    # Simulation parameters
    y0_init = 1.0      # Initial height
    v0_init = 0.0      # Initial velocity
    dt = 0.01          # Time step (s)
    sim_time = 2.0     # Total simulation time (s)
    steps = int(sim_time / dt)

    # Parameters
    params = [
        torch.tensor(9.81, requires_grad=True),  # g
    ]

    # Time steps and initial state
    ts = torch.linspace(0, sim_time, steps + 1)
    y0 = torch.tensor([y0_init, v0_init], dtype=torch.float64, requires_grad=True)

    # Simulate
    sim_time_start = time.time()
    yt = backward_euler_integrator(f, ts, y0, params=params)
    print("Implicit Euler time:", time.time() - sim_time_start)

    # Plotting
    t = ts.detach().numpy()
    yt_np = yt[:, 0].detach().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(t, yt_np, label="Backward Euler", linestyle="-", alpha=0.8, color="blue", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Height [m]")
    plt.title("Bouncing Particle Position vs. Time")
    plt.legend()
    plt.grid()
    plt.savefig('bouncing_particle.png')
    # plt.show() not used as per guidelines

    # Test gradients
    loss = torch.sum(yt[:, 0] ** 2)  # Loss: sum of squared positions

    grad_time_start = time.time()
    grads = torch.autograd.grad(
        outputs=loss,
        inputs=[params[0], y0],
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )
    print("Gradient computation time:", time.time() - grad_time_start)

    g_grad, y0_grad = grads
    print("Gradient w.r.t. g:", g_grad)
    print("Gradient w.r.t. y0:", y0_grad)

def numerical_gradient_test():
    # Simulation parameters
    y0_init = 1.0
    v0_init = 0.0
    dt = 0.01
    sim_time = 2.0
    steps = int(sim_time / dt)

    # Base parameters
    g_base = 9.81
    y0 = torch.tensor([y0_init, v0_init], dtype=torch.float64, requires_grad=True)
    ts = torch.linspace(0, sim_time, steps + 1)
    epsilon = 1e-4

    base_params = [torch.tensor(g_base, requires_grad=True)]

    sim_time_start = time.time()
    yt_base = backward_euler_integrator(f, ts, y0, params=base_params)
    print("Implicit Euler time:", time.time() - sim_time_start)

    loss = torch.sum(yt_base[:, 0] ** 2)
    grad_time_start = time.time()
    grads = torch.autograd.grad(
        outputs=loss,
        inputs=[base_params[0], y0],
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )
    print("Gradient computation time:", time.time() - grad_time_start)

    anal_g_grad, anal_y0_grad = grads

    # Numerical gradient for g
    with torch.no_grad():
        g_plus = torch.tensor(g_base + epsilon, dtype=torch.float64)
        yt_plus = backward_euler_integrator(f, ts, y0, params=[g_plus])
        loss_plus = torch.sum(yt_plus[:, 0] ** 2)

        g_minus = torch.tensor(g_base - epsilon, dtype=torch.float64)
        yt_minus = backward_euler_integrator(f, ts, y0, params=[g_minus])
        loss_minus = torch.sum(yt_minus[:, 0] ** 2)

        numerical_grad_g = (loss_plus - loss_minus) / (2 * epsilon)

    # Numerical gradient for y0[0] (initial position)
    with torch.no_grad():
        y0_plus = torch.tensor([y0_init + epsilon, v0_init], dtype=torch.float64)
        yt_plus = backward_euler_integrator(f, ts, y0_plus, params=[g_base])
        loss_plus = torch.sum(yt_plus[:, 0] ** 2)

        y0_minus = torch.tensor([y0_init - epsilon, v0_init], dtype=torch.float64)
        yt_minus = backward_euler_integrator(f, ts, y0_minus, params=[g_base])
        loss_minus = torch.sum(yt_minus[:, 0] ** 2)

        numerical_grad_y0 = (loss_plus - loss_minus) / (2 * epsilon)

    # Numerical gradient for y0[1] (initial velocity)
    with torch.no_grad():
        v0_plus = torch.tensor([y0_init, v0_init + epsilon], dtype=torch.float64)
        yt_plus = backward_euler_integrator(f, ts, v0_plus, params=[g_base])
        loss_plus = torch.sum(yt_plus[:, 0] ** 2)

        v0_minus = torch.tensor([y0_init, v0_init - epsilon], dtype=torch.float64)
        yt_minus = backward_euler_integrator(f, ts, v0_minus, params=[g_base])
        loss_minus = torch.sum(yt_minus[:, 0] ** 2)

        numerical_grad_v0 = (loss_plus - loss_minus) / (2 * epsilon)

    # Print results
    print(f"Analytical gradient for g: {anal_g_grad.item()}")
    print(f"Numerical gradient for g: {numerical_grad_g.item()}")
    print(f"Absolute difference for g: {abs(anal_g_grad.item() - numerical_grad_g.item())}")
    print()

    print(f"Analytical gradient for y0[0]: {anal_y0_grad[0].item()}")
    print(f"Numerical gradient for y0[0]: {numerical_grad_y0.item()}")
    print(f"Absolute difference for y0[0]: {abs(anal_y0_grad[0].item() - numerical_grad_y0.item())}")
    print()

    print(f"Analytical gradient for y0[1]: {anal_y0_grad[1].item()}")
    print(f"Numerical gradient for y0[1]: {numerical_grad_v0.item()}")
    print(f"Absolute difference for y0[1]: {abs(anal_y0_grad[1].item() - numerical_grad_v0.item())}")

if __name__ == "__main__":
    numerical_gradient_test()
    main()