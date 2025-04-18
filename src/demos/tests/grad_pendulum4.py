import time

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from xitorch.integrate import solve_ivp
from xitorch.optimize import rootfinder

torch.autograd.set_detect_anomaly(True)

# Define the dynamics function (correct as is)
def f(t: torch.Tensor, y: torch.Tensor, *params) -> torch.Tensor:
    g, l = params
    dv = -g * l * torch.sin(y[0])  # dv/dt
    dq = y[1]                      # dq/dt
    return torch.stack([dq, dv], dim=0)

def newton(fcn, x0, params, **config):
    tol=config["tol"]
    maxiter=config["maxiter"]
    x = x0
    for _ in range(maxiter):
        fx = fcn(x, *params)
        J = torch.autograd.functional.jacobian(lambda y: fcn(y, *params), x)
        delta = torch.linalg.solve(J, -fx)
        x = x + delta
        # if torch.norm(delta) < tol:
        #     break
    return x

def backward_euler_integrator(fcn, ts, y0, params, **kwargs):
    g, l = params
    n_steps = len(ts) - 1

    # Pre-allocate the full trajectory
    # Use a list instead of modifying a tensor in-place
    yt_list = [y0]  # Start with just the initial value

    for i in range(n_steps):
        t_curr, y_curr = ts[i], yt_list[i]
        t_next = ts[i + 1]
        dt = ts[i + 1] - ts[i]  # Calculate dt

        # Define the residual function for backward Euler
        def residual(y_next, y_curr, t_next, dt, g, l):
            return y_next - y_curr - dt * fcn(t_next, y_next, g, l)

        # Initial guess: forward Euler step
        y_guess = y_curr + dt * fcn(t_curr, y_curr, *params)

        # Find the next state
        y_next = rootfinder(
            residual,
            y_guess,
            method=newton,
            params=(y_curr, t_next, dt, g, l),
            tol=1e-9,
            maxiter=10
        )

        # Append to list instead of in-place modification
        yt_list.append(y_next)

    # Stack all results at the end
    yt = torch.stack(yt_list, dim=0)
    return yt


def main():
    # Simulation parameters
    q0 = 2.0           # initial position/angle
    u0 = 0.0           # initial velocity
    dt = 0.01         # time step (s)
    sim_time = 5.0     # total simulation time (s)
    steps = int(sim_time / dt)  # number of time steps

    # Parameters (with requires_grad=True for differentiation)
    params = [
        torch.tensor(9.81, requires_grad=True),  # g
        torch.tensor(1.0, requires_grad=True),   # l
    ]

    # Time steps and initial state
    ts = torch.linspace(0, sim_time, steps + 1)
    y0 = torch.tensor([q0, u0], dtype=torch.float64, requires_grad=True)

    # Solve the IVP
    sim_time = time.time()
    yt = backward_euler_integrator(
        f,
        ts,
        y0,
        method=backward_euler_integrator,
        params=params
    )
    print("Implicit Euler time:", time.time() - sim_time)

    # Plotting
    t = ts.detach().numpy()
    qt = yt[:, 0].detach().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(t, qt, label="Backward Euler", linestyle="-", alpha=0.8, color="red", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [rad]")
    plt.title("Position vs. Time")
    plt.legend()
    plt.grid()
    plt.show()

    # Optional: Test gradients (example)
    loss = torch.sum(yt[:, 0] ** 2)  # Example loss: trajectory squared

    # Compute gradients directly
    grad_time = time.time()
    grads = torch.autograd.grad(
        outputs=loss,
        inputs=[params[0], params[1], y0],  # List the input parameters you want gradients for
        create_graph=True,  # Only use this if you need higher-order gradients
        retain_graph=True,  # Keeps the computation graph in memory
        allow_unused=True  # Doesn't error if some inputs don't affect the output
    )
    print("Gradient computation time:", time.time() - grad_time)

    # Extract the gradients
    g_grad, l_grad, y0_grad = grads
    print("Gradient w.r.t. g:", g_grad)
    print("Gradient w.r.t. l:", l_grad)
    print("Gradient w.r.t. y0:", y0_grad)
    
    """
    Backward Euler time: 17.51150870323181
    Gradient computation time: 208.47105026245117
    Gradient w.r.t. g: tensor(-11.0353, grad_fn=<AddBackward0>)
    Gradient w.r.t. l: tensor(-108.2564, grad_fn=<AddBackward0>)
    Gradient w.r.t. y0: tensor([77.5622,  5.1293], dtype=torch.float64, grad_fn=<AddBackward0>)
    
    
    Implicit Euler time: 22.91762614250183
    Gradient computation time: 6.0032126903533936
    Gradient w.r.t. g: tensor(-1.8999, grad_fn=<AddBackward0>)
    Gradient w.r.t. l: tensor(-18.6379, grad_fn=<AddBackward0>)
    Gradient w.r.t. y0: tensor([11.4184,  0.7269], dtype=torch.float64, grad_fn=<AddBackward0>)
    """


def numerical_gradient_test():
    # Simulation parameters
    q0 = 2.0  # initial position/angle
    u0 = 0.0  # initial velocity
    dt = 0.01  # time step (reduced to speed up the test)
    sim_time = 5.0  # reduced simulation time for testing
    steps = int(sim_time / dt)

    # Base parameters
    g_base = 9.81
    l_base = 1.0
    y0 = torch.tensor([q0, u0], dtype=torch.float64, requires_grad=True)
    ts = torch.linspace(0, sim_time, steps + 1)

    # Small perturbation for finite difference
    epsilon = 1e-4

    # Parameters (with requires_grad=True for differentiation)
    base_params = [
        torch.tensor(9.81, requires_grad=True),  # g
        torch.tensor(1.0, requires_grad=True),  # l
    ]

    sim_time = time.time()
    yt_base = backward_euler_integrator(
        f,
        ts,
        y0,
        params=base_params
    )
    print("Implicit Euler time:", time.time() - sim_time)

    loss = torch.sum(yt_base[:, 0] ** 2)  # Example loss: trajectory squared

    # Get analytical gradient using autograd
    grad_time = time.time()
    grads = torch.autograd.grad(
        outputs=loss,
        inputs=[base_params[0], base_params[1], y0],  # List the input parameters you want gradients for
        create_graph=True,  # Only use this if you need higher-order gradients
        retain_graph=True,  # Keeps the computation graph in memory
        allow_unused=True  # Doesn't error if some inputs don't affect the output
    )
    print("Gradient computation time:", time.time() - grad_time)

    # Extract the gradients
    anal_g_grad, anal_l_grad, anal_y0_grad = grads

    # Get the numerical gradient for g using finite differences
    with torch.no_grad():
        g_plus = torch.tensor(g_base + epsilon, dtype=torch.float64)
        params_plus = [g_plus, l_base]
        yt_plus = backward_euler_integrator(f, ts, y0, params=params_plus)
        loss_plus = torch.sum(yt_base[:, 0] ** 2)

        g_minus = torch.tensor(g_base - epsilon, dtype=torch.float64)
        params_minus = [g_minus, l_base]
        yt_minus = backward_euler_integrator(f, ts, y0, params=params_minus)
        loss_minus = torch.sum(yt_minus[:, 0] ** 2)

        numerical_grad_g = (loss_plus - loss_minus) / (2 * epsilon)

    # Get the numerical gradient for l using finite differences
    with torch.no_grad():
        l_plus = torch.tensor(l_base + epsilon, dtype=torch.float64)
        params_plus = [g_base, l_plus]
        yt_plus = backward_euler_integrator(f, ts, y0, params=params_plus)
        loss_plus = torch.sum(yt_plus[:, 0] ** 2)

        l_minus = torch.tensor(l_base - epsilon, dtype=torch.float64)
        params_minus = [g_base, l_minus]
        yt_minus = backward_euler_integrator(f, ts, y0, params=params_minus)
        loss_minus = torch.sum(yt_minus[:, 0] ** 2)

        numerical_grad_l = (loss_plus - loss_minus) / (2 * epsilon)

    # Get the numerical gradient for y0 using finite differences
    with torch.no_grad():
        # Perturb q0
        q0_plus = torch.tensor([q0 + epsilon, u0], dtype=torch.float64)
        params_plus = [g_base, l_base]
        yt_plus = backward_euler_integrator(f, ts, q0_plus, params=params_plus)
        loss_plus = torch.sum(yt_plus[:, 0] ** 2)

        q0_minus = torch.tensor([q0 - epsilon, u0], dtype=torch.float64)
        params_minus = [g_base, l_base]
        yt_minus = backward_euler_integrator(f, ts, q0_minus, params=params_minus)
        loss_minus = torch.sum(yt_minus[:, 0] ** 2)

        numerical_grad_q0 = (loss_plus - loss_minus) / (2 * epsilon)

        # Perturb u0
        u0_plus = torch.tensor([q0, u0 + epsilon], dtype=torch.float64)
        params_plus = [g_base, l_base]
        yt_plus = backward_euler_integrator(f, ts, u0_plus, params=params_plus)
        loss_plus = torch.sum(yt_plus[:, 0] ** 2)

        u0_minus = torch.tensor([q0, u0 - epsilon], dtype=torch.float64)
        params_minus = [g_base, l_base]
        yt_minus = backward_euler_integrator(f, ts, u0_minus, params=params_minus)
        loss_minus = torch.sum(yt_minus[:, 0] ** 2)
        numerical_grad_u0 = (loss_plus - loss_minus) / (2 * epsilon)

    # Print results
    print(f"Analytical gradient for g: {anal_g_grad.item()}")
    print(f"Numerical gradient for g: {numerical_grad_g.item()}")
    print(f"Absolute difference for g: {abs(anal_g_grad.item() - numerical_grad_g.item())}")
    print(f"Relative difference for g: {abs(anal_g_grad.item() - numerical_grad_g.item()) / max(abs(anal_g_grad.item()), abs(numerical_grad_g.item()))}")
    print()

    print(f"Analytical gradient for l: {anal_l_grad.item()}")
    print(f"Numerical gradient for l: {numerical_grad_l.item()}")
    print(f"Absolute difference for l: {abs(anal_l_grad.item() - numerical_grad_l.item())}")
    print(f"Relative difference for l: {abs(anal_l_grad.item() - numerical_grad_l.item()) / max(abs(anal_l_grad.item()), abs(numerical_grad_l.item()))}")
    print()

    print(f"Analytical gradient for q0: {anal_y0_grad[0].item()}")
    print(f"Numerical gradient for q0: {numerical_grad_q0.item()}")
    print(f"Absolute difference for q0: {abs(anal_y0_grad[0].item() - numerical_grad_q0.item())}")
    print(f"Relative difference for q0: {abs(anal_y0_grad[0].item() - numerical_grad_q0.item()) / max(abs(anal_y0_grad[0].item()), abs(numerical_grad_q0.item()))}")
    print()

    print(f"Analytical gradient for u0: {anal_y0_grad[1].item()}")
    print(f"Numerical gradient for u0: {numerical_grad_u0.item()}")
    print(f"Absolute difference for u0: {abs(anal_y0_grad[1].item() - numerical_grad_u0.item())}")
    print(f"Relative difference for u0: {abs(anal_y0_grad[1].item() - numerical_grad_u0.item()) / max(abs(anal_y0_grad[1].item()), abs(numerical_grad_u0.item()))}")
    print()

    """
    Gradient computation time: 169.0160505771637
    Analytical gradient for g: -11.035279273986816
    Numerical gradient for g: -10.988046126119144
    Absolute difference for g: 0.04723314786767219
    Relative difference for g: 0.004280195063029686

    Analytical gradient for l: -108.2563705444336
    Numerical gradient for l: -108.26196967798296
    Absolute difference for l: 0.005599133549367252
    Relative difference for l: 5.1718378725432866e-05

    Analytical gradient for q0: 77.5622071625318
    Numerical gradient for q0: 77.58781326920428
    Absolute difference for q0: 0.025606106672469764
    Relative difference for q0: 0.0003300274307722138

    Analytical gradient for u0: 5.12931281644632
    Numerical gradient for u0: 5.062601606624639
    Absolute difference for u0: 0.06671120982168066
    Relative difference for u0: 0.013005876656183232
    """


if __name__ == "__main__":
    numerical_gradient_test()
    main()