import time
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from xitorch.integrate import solve_ivp
from xitorch.optimize import rootfinder

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


# Custom backward Euler integrator using rootfinder
def backward_euler_integrator(fcn, ts, y0, params, **kwargs):
    g, l = params
    yt_list = [y0]  # Start with just the initial value

    for i in range(len(ts) - 1):
        t_curr, y_curr = ts[i], yt_list[i]
        t_next = ts[i + 1]
        dt = ts[i + 1] - ts[i]  # Calculate dt

        # Define the residual function for backward Euler
        def residual(y_next, y_curr, t_next, dt, g, l):
            return y_next - y_curr - dt * fcn(t_next, y_next, g, l)

        # Initial guess: forward Euler step
        y_guess = y_curr + dt * fcn(t_curr, y_curr, *params)

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
    yt = solve_ivp(
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
    loss = torch.sum(yt[:, 0] ** 2)  # Example loss: final state squared
    
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
    Implicit Euler time: 3.306675672531128
    Gradient computation time: 225.54714179039001
    Gradient w.r.t. g: tensor(-1.8337, grad_fn=<ToCopyBackward0>)
    Gradient w.r.t. l: tensor(-17.9882, grad_fn=<ToCopyBackward0>)
    Gradient w.r.t. y0: tensor([11.1367,  0.8249], dtype=torch.float64, grad_fn=<AddBackward0>)
    """

if __name__ == "__main__":
    main()