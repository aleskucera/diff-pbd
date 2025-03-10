import torch
import matplotlib
matplotlib.use('TkAgg')

# Parameters
g = 9.81  # acceleration due to gravity (m/s^2)
l = 1.0   # length of the pendulum (m)
dt = 0.01  # time step (s)
num_steps = 1000  # number of steps

# Initial conditions
q = torch.tensor([0.5], dtype=torch.float32, requires_grad=True)  # initial angle (rad)
v = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)  # initial angular velocity (rad/s)

# Function to solve the implicit equation using Newton's method
def solve_implicit(q_k, v_k):
    q_next = q_k.clone().detach().requires_grad_(True)
    v_next = v_k.clone().detach().requires_grad_(True)

    for _ in range(10):  # Newton's iterations
        # Implicit Euler equations
        f1 = q_next - q_k - dt * v_next
        f2 = v_next - v_k + dt * (g / l) * torch.sin(q_next)

        # Jacobian matrix
        J11 = torch.autograd.grad(f1, q_next, retain_graph=True)[0]
        J12 = torch.autograd.grad(f1, v_next, retain_graph=True)[0]
        J21 = torch.autograd.grad(f2, q_next, retain_graph=True)[0]
        J22 = torch.autograd.grad(f2, v_next, retain_graph=True)[0]
        J = torch.stack([torch.cat([J11, J12]), torch.cat([J21, J22])])

        # Function values
        F = torch.stack([f1, f2])

        # Update
        delta = torch.linalg.solve(J, -F)
        q_next = q_next + delta[0]
        v_next = v_next + delta[1]

    return q_next.detach(), v_next.detach()

# Simulate the pendulum
qs, vs = [], []
for step in range(num_steps):
    q, v = solve_implicit(q, v)
    qs.append(q.item())
    vs.append(v.item())

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

# Phase space plot
plt.subplot(1, 2, 1)
plt.plot(qs, vs)
plt.xlabel('Angle (rad)')
plt.ylabel('Angular velocity (rad/s)')
plt.title('Phase Space (Backward Euler)')

# Angle vs time
plt.subplot(1, 2, 2)
plt.plot([dt * i for i in range(num_steps)], qs)
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Angle vs Time (Backward Euler)')

plt.tight_layout()
plt.show()
