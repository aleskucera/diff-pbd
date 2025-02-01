import time

import torch


def forward_sensitivities(A: torch.Tensor, del_J__del_theta: torch.Tensor,
                          del_J__del_x: torch.Tensor,
                          d_b__d_theta: torch.Tensor):
    d_x__d_theta = torch.linalg.solve(A, d_b__d_theta)
    d_j__d_theta__forward = del_J__del_theta + del_J__del_x.T @ d_x__d_theta
    return d_j__d_theta__forward


def adjoint_sensitivities(A: torch.Tensor, del_J__del_theta: torch.Tensor,
                          del_J__del_x: torch.Tensor,
                          d_b__d_theta: torch.Tensor):
    adjoint_variable = torch.linalg.solve(A.T, del_J__del_x)
    d_j__d_theta__adjoint = del_J__del_theta + adjoint_variable.T @ d_b__d_theta
    return d_j__d_theta__adjoint


if __name__ == "__main__":
    A = torch.tensor([[10, 2, 1], [2, 5, 1], [1, 1, 3]], dtype=torch.float32)

    # -----------------  Creating a reference solution  -----------------
    b_true = torch.tensor([5, 4, 3], dtype=torch.float32).view(-1, 1)
    x_ref = torch.linalg.solve(A, b_true)

    # ----------------- [A] Solve the classical system -----------------
    b_guess = torch.tensor([1, 1, 1], dtype=torch.float32).view(-1, 1)
    x = torch.linalg.solve(A, b_guess)
    J = 0.5 * torch.sum((x - x_ref)**2)
    print(f"J = {J}")

    # ----------------- [B] Obtaining gradients -----------------
    del_J__del_theta = torch.zeros((1, 3))  # (1, 3)
    del_J__del_x = (x - x_ref)  # (3, 1)
    d_b__d_theta = torch.eye(3)  # (3, 3)

    # [1] Finite Difference Method
    start_time = time.time()
    eps = 1.0e-3
    d_J__d_theta_finite_difference = torch.zeros((1, 3))
    for i in range(3):
        b_augmented = b_guess.clone()
        b_augmented[i] += eps

        x_augmented = torch.linalg.solve(A, b_augmented)
        J_augmented = 0.5 * (x_augmented - x_ref).T @ (x_augmented - x_ref)

        d_J__d_theta_finite_difference[0, i] = (J_augmented - J) / eps
    finite_difference_time = time.time() - start_time
    print(f"dJ/dtheta (finite difference) = {d_J__d_theta_finite_difference}")
    print(
        f"Time taken by finite difference method: {finite_difference_time:.6f} seconds"
    )

    # [2] Forward Method
    start_time = time.time()
    d_j__d_theta__forward = forward_sensitivities(A, del_J__del_theta,
                                                  del_J__del_x, d_b__d_theta)
    forward_time = time.time() - start_time
    print(f"dJ/dtheta (forward method) = {d_j__d_theta__forward}")
    print(f"Time taken by forward method: {forward_time:.6f} seconds")

    # [3] Adjoint Method
    start_time = time.time()
    d_j__d_theta__adjoint = adjoint_sensitivities(A, del_J__del_theta,
                                                  del_J__del_x, d_b__d_theta)
    adjoint_time = time.time() - start_time
    print(f"dJ/dtheta (adjoint method) = {d_j__d_theta__adjoint}")
    print(f"Time taken by adjoint method: {adjoint_time:.6f} seconds")

    # [4] PyTorch Autograd
    start_time = time.time()

    # Create tensors with requires_grad=True for parameters we want to differentiate with respect to
    b_autograd = torch.tensor([1, 1, 1],
                              dtype=torch.float32,
                              requires_grad=True).view(-1, 1)
    # Make sure that the b_autograd retains the gradients
    b_autograd.retain_grad()

    # Forward pass
    x_autograd = torch.linalg.solve(A, b_autograd)
    J_autograd = 0.5 * torch.sum((x_autograd - x_ref)**2)

    # Backward pass
    J_autograd.backward(retain_graph=True)

    # Get gradients
    d_j__d_theta_autograd = b_autograd.grad.T

    autograd_time = time.time() - start_time
    print(f"dJ/dtheta (autograd) = {d_j__d_theta_autograd}")
    print(f"Time taken by autograd method: {autograd_time:.6f} seconds")
