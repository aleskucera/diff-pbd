import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from xitorch.integrate import solve_ivp

def f(t: torch.Tensor, y: torch.Tensor, *params) -> torch.Tensor:
    g, l = params
    dv = -g * l * torch.sin(y[0])
    dq = y[1]
    return torch.stack([dq, dv], dim=0)

def backward_euler_integrator(fcn, ts, y0, params, **kwargs):
    dt = ts[1] - ts[0]  # assuming uniform time steps
    yt = torch.empty((len(ts), *y0.shape), dtype=y0.dtype, device=y0.device)
    yt[0] = y0

    # Maximum iterations and convergence tolerance
    max_iter = 30
    tol = 1e-9

    for i in tqdm(range(len(ts) - 1)):
        t_curr, y_curr = ts[i], yt[i]
        t_next = ts[i + 1]

        # Initial guess: use forward Euler as starting point
        y_guess = y_curr + dt * fcn(t_curr, y_curr, *params)

        # Newton iterations
        y_next = y_guess.clone()

        for j in range(max_iter):
            # Compute residual: y_next - y_curr - dt * f(t_next, y_next)
            f_next = fcn(t_next, y_next, *params)
            residual = y_next - y_curr - dt * f_next

            # Check for convergence
            if torch.all(torch.abs(residual) < tol):
                # print(f"Step {i}: Converged after {j + 1} iterations.")
                break

            # Compute Jacobian approximation for the system
            # For a 2D system (like our oscillator or bounce), we can manually compute it
            epsilon = 1e-6

            # Perturb position
            y_perturbed = y_next.clone()
            y_perturbed[0] += epsilon
            f_perturbed = fcn(t_next, y_perturbed, *params)
            df_dq = (f_perturbed - f_next) / epsilon

            # Perturb velocity
            y_perturbed = y_next.clone()
            y_perturbed[1] += epsilon
            f_perturbed = fcn(t_next, y_perturbed, *params)
            df_dv = (f_perturbed - f_next) / epsilon

            # Construct the Jacobian: I - dt * df/dy
            J = torch.empty((2, 2), dtype=y_next.dtype, device=y_next.device)
            J[0, 0] = 1 - dt * df_dq[0]  # df_q/dq
            J[1, 0] = dt * df_dv[0]  # df_q/dv
            J[0, 1] = dt * df_dq[1]  # df_v/dq
            J[1, 1] = 1 - dt * df_dv[1]  # df_v/dv

            # Newton update: y_next = y_next - J^-1 * residual
            # Solve linear system J * delta = residual
            # For 2x2 system, we can just invert directly
            det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            J_inv = torch.empty((2, 2), dtype=y_next.dtype, device=y_next.device)
            J_inv[0, 0] = J[1, 1] / det
            J_inv[0, 1] = -J[0, 1] / det
            J_inv[1, 0] = -J[1, 0] / det
            J_inv[1, 1] = J[0, 0] / det

            # Compute update
            delta = torch.matmul(J_inv, residual)
            y_next = y_next - delta

            # Additional termination condition for very small updates
            if torch.all(torch.abs(delta) < tol):
                print(f"Converged after {j} iterations.")
                break

        yt[i + 1] = y_next

    return yt

@torch.no_grad()
def numerical_gradient(ts: torch.Tensor, y0: torch.Tensor, params: list):
    raise NotImplementedError("Numerical gradient calculation is not implemented for now.")


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
    ]

    # Create time steps tensor
    ts = torch.linspace(0, sim_time, steps + 1)
    y0 = torch.tensor([q0, u0], dtype=torch.float64, requires_grad=True)  # initial state

    backward_results = solve_ivp(
        f,
        ts,
        y0,
        method=backward_euler_integrator,
        params=params
    )

    # Convert to numpy for plotting
    time = ts.detach().numpy()
    backward_q = backward_results[:, 0].detach().numpy()

    # Plot positions over time
    plt.figure(figsize=(10, 6))
    plt.plot(time, backward_q, label="Backward Euler", linestyle="-", alpha=0.8, color="red", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [rad]")
    plt.title(f"Position vs. Time")
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    main()