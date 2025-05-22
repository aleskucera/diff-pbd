import torch
import time
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Parameters
iteration_counts = [100, 1000, 5000, 10000, 100000]  # Different iteration counts to test
runs_per_case = 5  # Number of runs for averaging
x_initial = torch.tensor([1.0], requires_grad=True)  # Initial input
w = torch.tensor([0.5], requires_grad=True)  # Parameter
loss_fn = lambda x: x ** 2  # Simple loss function (squared value)

# Function to run one iteration
def one_iteration(x, w):
    return x * w

# Case 1: No autograd
def case1_no_autograd(n_iterations):
    x = x_initial.clone().detach()  # No grad tracking
    w_detach = w.clone().detach()
    outputs = []
    for _ in range(n_iterations):
        x = one_iteration(x, w_detach)
        outputs.append(x)
    loss = sum(loss_fn(out) for out in outputs)
    return loss

# Case 2: Gradient w.r.t. initial input
def case2_grad_input(n_iterations):
    x = x_initial
    outputs = []
    for _ in range(n_iterations):
        x = one_iteration(x, w)
        outputs.append(x)
    loss = sum(loss_fn(out) for out in outputs)
    loss.backward()  # Compute gradients
    grad_x = x_initial.grad.clone()  # Save gradient
    x_initial.grad.zero_()  # Clear gradients
    return grad_x

# Case 3: Gradient w.r.t. parameter
def case3_grad_param(n_iterations):
    x = x_initial.clone().detach()  # Detach input to focus on w's grad
    outputs = []
    for _ in range(n_iterations):
        x = one_iteration(x, w)
        outputs.append(x)
    loss = sum(loss_fn(out) for out in outputs)
    loss.backward()  # Compute gradients
    grad_w = w.grad.clone()  # Save gradient
    w.grad.zero_()  # Clear gradients
    return grad_w

# Timing function
def measure_time(fn, n_iterations):
    times = []
    for _ in range(runs_per_case):
        start = time.perf_counter()
        result = fn(n_iterations)
        end = time.perf_counter()
        times.append(end - start)
    return sum(times) / runs_per_case

# Collect timing data
times_case1 = []
times_case2 = []
times_case3 = []

for n in iteration_counts:
    print(f"Testing with {n} iterations...")
    times_case1.append(measure_time(case1_no_autograd, n))
    times_case2.append(measure_time(case2_grad_input, n))
    times_case3.append(measure_time(case3_grad_param, n))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(iteration_counts, times_case1, 'b-o', label='No autograd')
plt.plot(iteration_counts, times_case2, 'r-o', label='Gradient w.r.t. input')
plt.plot(iteration_counts, times_case3, 'g-o', label='Gradient w.r.t. parameter')
plt.title('Execution Time vs. Number of Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('Average Execution Time (seconds)')
plt.grid(True)
plt.legend()

# Save the plot
plt.savefig('execution_times.png')