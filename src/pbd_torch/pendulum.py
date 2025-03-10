import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

import matplotlib
matplotlib.use('TkAgg')


# Define the simple pendulum system parameters
g = 9.81  # acceleration due to gravity (m/s^2)
l = 1.0  # length of the pendulum (m)
m = 1.0  # mass of the pendulum bob (kg)

# Define the force acting on the bounce
k = 20000.0  # spring constant (N/m)

EXPERIMENT = "bounce"  # or "bounce"

def accel_pendulum(q: torch.Tensor) -> torch.Tensor:
    """Compute the force acting on the pendulum."""
    return -g * l * torch.sin(q)

def accel_bounce(q: torch.Tensor) -> torch.Tensor:
    """Compute the force acting on the bounce."""
    if q > 0:
        return torch.tensor([-g])
    else:
        return -k * q / m - g

def forward_euler(q0: torch.Tensor, u0: torch.Tensor, dt: float, num_steps: int):
    history = {"t": [0.0],
               "q": [q0.item()],
               "u": [u0.item()]}

    q = q0.clone()
    u = u0.clone()
    for _ in range(num_steps):
        a = -(g / l) * torch.sin(q)
        u_next = u + dt * a
        q_next = q + dt * u

        history["q"].append(q_next.item())
        history["u"].append(u_next.item())
        history["t"].append(history["t"][-1] + dt)

        q = q_next
        u = u_next

    return history

def symplectic_euler(q0: torch.Tensor, u0: torch.Tensor, dt: float, num_steps: int):
    history = {"t": [0.0],
               "q": [q0.item()],
               "u": [u0.item()]}

    if EXPERIMENT == "pendulum":
        accel = accel_pendulum
    elif EXPERIMENT == "bounce":
        accel = accel_bounce

    q = q0.clone()
    u = u0.clone()
    for _ in range(num_steps):
        # Compute the next state using the symplectic Euler method
        a = accel(q)
        u_next = u + dt * a
        q_next = q + dt * u_next

        history["q"].append(q_next.item())
        history["u"].append(u_next.item())
        history["t"].append(history["t"][-1] + dt)

        q = q_next
        u = u_next

    return history

def symplectic_euler_position(q0: torch.Tensor, u0: torch.Tensor, dt: float, num_steps: int):
    history = {"t": [0.0],
               "q": [q0.item()],
               "u": [u0.item()]}

    q = q0.clone()
    u = u0.clone()
    for _ in range(num_steps):
        a = -(g / l) * torch.sin(q)
        q_next = q + dt * u + dt**2 * a
        u_next = (q_next - q) / dt

        history["q"].append(q_next.item())
        history["u"].append(u_next.item())
        history["t"].append(history["t"][-1] + dt)

        q = q_next
        u = u_next

    return history



def backward_euler(q0: torch.Tensor, u0: torch.Tensor, dt: float, num_steps: int):
    history = {"t": [0.0],
               "q": [q0.item()],
               "u": [u0.item()]}

    q = q0.clone().item()
    u = u0.clone().item()

    def residual(x: list, q_prev: float, u_prev: float) -> list:
        """Compute the residual of the implicit system."""
        q_next, u_next = x
        r_q = q_next - q_prev - dt * u_next
        r_u = u_next - u_prev - dt * (-g / l * np.sin(q_next))
        return [r_q, r_u]

    for _ in range(num_steps):
        # Initial guess for the solver
        x0 = [q, u]

        # Use scipy.optimize.root to solve the implicit system
        sol = root(residual, x0, args=(q, u))

        if sol.success:
            q_next, u_next = sol.x
        else:
            raise RuntimeError("Root-finding failed: " + sol.message)

        history["q"].append(q_next)
        history["u"].append(u_next)
        history["t"].append(history["t"][-1] + dt)

        # Update current state
        q, u = q_next, u_next

    return history


def rk4(q0: torch.Tensor, u0: torch.Tensor, dt: float, num_steps: int):
    history = {"t": [0.0],
               "q": [q0.item()],
               "u": [u0.item()]}

    if EXPERIMENT == "pendulum":
        accel = accel_pendulum
    elif EXPERIMENT == "bounce":
        accel = accel_bounce

    q = q0.clone()
    u = u0.clone()
    for _ in range(num_steps):
        k1_q = dt * u
        k1_u = dt * accel(q)

        k2_q = dt * (u + 0.5 * k1_u)
        k2_u = dt * accel(q + 0.5 * k1_q)

        k3_q = dt * (u + 0.5 * k2_u)
        k3_u = dt * accel(q + 0.5 * k2_q)

        k4_q = dt * (u + k3_u)
        k4_u = dt * accel(q + k3_q)

        q_next = q + (k1_q + 2 * k2_q + 2 * k3_q + k4_q) / 6
        u_next = u + (k1_u + 2 * k2_u + 2 * k3_u + k4_u) / 6

        # k1_q = dt * u
        # k1_u = dt * (-(g / l) * torch.sin(q))
        #
        # k2_q = dt * (u + 0.5 * k1_u)
        # k2_u = dt * (-(g / l) * torch.sin(q + 0.5 * k1_q))
        #
        # k3_q = dt * (u + 0.5 * k2_u)
        # k3_u = dt * (-(g / l) * torch.sin(q + 0.5 * k2_q))
        #
        # k4_q = dt * (u + k3_u)
        # k4_u = dt * (-(g / l) * torch.sin(q + k3_q))
        #
        # q_next = q + (k1_q + 2 * k2_q + 2 * k3_q + k4_q) / 6
        # u_next = u + (k1_u + 2 * k2_u + 2 * k3_u + k4_u) / 6

        history["q"].append(q_next.item())
        history["u"].append(u_next.item())
        history["t"].append(history["t"][-1] + dt)

        q = q_next
        u = u_next

    return history

def main():
    # Simulation parameters
    # q0 = 0.2  # initial angle (rad)
    q0 = 2.0 # height of the bounce
    u0 = 0.0  # initial angular velocity (rad/s)
    dt = 0.01  # time step (s)
    sim_time = 10.0  # total simulation time (s)
    steps = int(sim_time / dt)  # number of time steps

    # Run the simulation
    forward_history = forward_euler(torch.tensor([q0]), torch.tensor([u0]), dt, steps)
    symplectic_history = symplectic_euler(torch.tensor([q0]), torch.tensor([u0]), dt, steps)
    symplectic_pos_history = symplectic_euler_position(torch.tensor([q0]), torch.tensor([u0]), dt, steps)
    backward_history = backward_euler(torch.tensor([q0]), torch.tensor([u0]), dt, steps)
    rk4_history = rk4(torch.tensor([q0]), torch.tensor([u0]), dt, steps)

    # Extract results
    forward_q = np.array(forward_history["q"])
    forward_u = np.array(forward_history["u"])
    forward_t = np.array(forward_history["t"])

    symplectic_q = np.array(symplectic_history["q"])
    symplectic_u = np.array(symplectic_history["u"])
    symplectic_t = np.array(symplectic_history["t"])

    symplectic_pos_q = np.array(symplectic_pos_history["q"])
    symplectic_pos_u = np.array(symplectic_pos_history["u"])
    symplectic_pos_t = np.array(symplectic_pos_history["t"])

    backward_q = np.array(backward_history["q"])
    backward_u = np.array(backward_history["u"])
    backward_t = np.array(backward_history["t"])

    rk4_q = np.array(rk4_history["q"])
    rk4_u = np.array(rk4_history["u"])
    rk4_t = np.array(rk4_history["t"])

    assert np.allclose(forward_t, symplectic_t)
    assert np.allclose(symplectic_t, symplectic_pos_t)
    assert np.allclose(symplectic_pos_t, backward_t)
    assert np.allclose(backward_t, rk4_t)
    time = forward_t

    # # Create a phase space plot (q vs v)
    # plt.figure(figsize=(8, 6))
    # # plt.plot(forward_q, forward_u,label="Forward Euler", linestyle="-", alpha=0.8, color="blue")
    # plt.plot(symplectic_q, symplectic_u, label="Symplectic Euler", linestyle="-", alpha=1.0, color="orange")
    # plt.plot(symplectic_pos_q, symplectic_pos_u, label="Symplectic Euler (Position)", linestyle="-", alpha=1.0, color="green")
    # # plt.plot(backward_q, backward_u, label="Backward Euler", linestyle="-", alpha=0.8, color="red")
    # plt.plot(rk4_q, rk4_u, label="RK4", linestyle=":", alpha=0.8, color="black")
    # plt.xlabel("Angle q (rad)")
    # plt.ylabel("Angular velocity v (rad/s)")
    # plt.title("Phase Space of Simple Pendulum (Forward Euler Method)")
    # plt.legend()
    # plt.grid()
    # plt.show()

    # Plot positions over time
    plt.figure(figsize=(10, 6))
    # plt.plot(time, forward_q, label="Forward Euler", linestyle="-", alpha=0.8, color="blue", linewidth=2)
    plt.plot(time, symplectic_q, label="Symplectic Euler", linestyle="-", alpha=1.0, color="orange", linewidth=2)
    # plt.plot(time, symplectic_pos_q, label="Symplectic Euler (Position)", linestyle="-", alpha=1.0, color="green", linewidth=2)
    # plt.plot(time, backward_q, label="Backward Euler", linestyle="-", alpha=0.8, color="red", linewidth=2)
    plt.plot(time, rk4_q, label="RK4", linestyle=":", alpha=0.8, color="black", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Height [m]")
    # plt.title("Position vs. Time for Different Integrators Methods")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()