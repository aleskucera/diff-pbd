import time

import torch


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication for batched input."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def compute_H(q: torch.Tensor) -> torch.Tensor:
    """Compute the H matrix for batched quaternions."""
    w, x, y, z = q.unbind(dim=-1)
    H = 0.5 * torch.stack(
        [
            torch.stack([-x, -y, -z], dim=-1),
            torch.stack([w, -z, y], dim=-1),
            torch.stack([z, w, -x], dim=-1),
            torch.stack([-y, x, w], dim=-1),
        ],
        dim=-2,
    )
    return H


def integrate_angle(q: torch.Tensor, w: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    theta = torch.norm(w, dim=-1, keepdim=True) * dt  # Rotation angle
    axis = w / (torch.norm(w, dim=-1, keepdim=True) + 1e-10)  # Rotation axis
    wx, wy, wz = axis.unbind(dim=-1)
    half_theta = theta / 2
    sin_half_theta = torch.sin(half_theta)
    cos_half_theta = torch.cos(half_theta)

    q_delta = torch.stack(
        [
            cos_half_theta.squeeze(-1),
            wx * sin_half_theta.squeeze(-1),
            wy * sin_half_theta.squeeze(-1),
            wz * sin_half_theta.squeeze(-1),
        ],
        dim=-1,
    )

    q_new = quat_mul(q, q_delta)
    return q_new / torch.norm(q_new, dim=-1, keepdim=True)


def integrate_angle_approx(
    q: torch.Tensor, w: torch.Tensor, dt: torch.Tensor
) -> torch.Tensor:
    q_delta = (
        0.5
        * quat_mul(torch.cat([torch.zeros_like(w[..., :1]), w], dim=-1), q)
        * dt.unsqueeze(-1)
    )
    q_new = q + q_delta
    return q_new / torch.norm(q_new, dim=-1, keepdim=True)


def integrate_quaternion_discretized(
    q: torch.Tensor, w: torch.Tensor, dt: torch.Tensor
) -> torch.Tensor:
    H = compute_H(q)
    q_new = q + (dt.unsqueeze(-1).unsqueeze(-1) * H @ w.unsqueeze(-1)).squeeze(-1)
    return q_new / torch.norm(q_new, dim=-1, keepdim=True)


def angular_difference(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
    q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)
    dot_product = torch.sum(q1 * q2, dim=-1).clamp(-1.0, 1.0)
    return 2 * torch.acos(torch.abs(dot_product))


def test_integration_methods(q, w, dt, num_iterations=1000):
    results = {}

    start_time = time.time()
    for _ in range(num_iterations):
        q_exact = integrate_angle(q, w, dt)
    results["Exact Integration Time"] = time.time() - start_time

    start_time = time.time()
    for _ in range(num_iterations):
        q_approx = integrate_angle_approx(q, w, dt)
    results["Approximate Integration Time"] = time.time() - start_time

    start_time = time.time()
    for _ in range(num_iterations):
        q_discretized = integrate_quaternion_discretized(q, w, dt)
    results["Discretized Integration Time"] = time.time() - start_time

    results["Approx vs Exact Angular Error"] = (
        angular_difference(q_approx, q_exact).max().item()
    )
    results["Discretized vs Exact Angular Error"] = (
        angular_difference(q_discretized, q_exact).max().item()
    )
    results["Approx vs Discretized Angular Error"] = (
        angular_difference(q_approx, q_discretized).max().item()
    )

    return results


# Generate 100 random test cases
num_cases = 100
batch_size = 100

# Generate random normalized quaternions
q_random = torch.randn((batch_size, 4), device="cuda")
q_random /= torch.norm(q_random, dim=-1, keepdim=True)

# Generate random angular velocities (w)
w_random = torch.randn((batch_size, 3), device="cuda")

# Generate random time steps
dt_random = torch.rand((batch_size,), device="cuda") * 0.1  # Random dt in [0, 0.1]

# Run the test
print("Randomized Test Results:")
random_results = test_integration_methods(q_random, w_random, dt_random)
for key, value in random_results.items():
    if "Time" in key:
        print(f"  {key}: {value:.6f} seconds")
    else:
        print(f"  {key}: {value:.6f} radians")
