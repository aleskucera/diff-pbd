import time

import torch
from pbd_torch.integration import integrate_bodies_vectorized
from pbd_torch.integration import integrate_body
from pbd_torch.transform import normalize_quat


def test_integration_equivalence():
    """Test that vectorized integration produces identical results to the original function."""

    print("Testing integration equivalence...")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test with different numbers of bodies
    for num_bodies in [1, 2, 5, 10]:
        print(f"\nTesting with {num_bodies} bodies")

        # Create random test data
        body_q = torch.rand((num_bodies, 7), dtype=torch.float32)
        # Normalize quaternions in the input
        for i in range(num_bodies):
            body_q[i, 3:] = normalize_quat(body_q[i, 3:])

        body_qd = (
            torch.rand((num_bodies, 6), dtype=torch.float32) * 2 - 1
        )  # Range [-1, 1]
        body_f = (
            torch.rand((num_bodies, 6), dtype=torch.float32) * 10 - 5
        )  # Range [-5, 5]
        body_inv_mass = (
            torch.rand(num_bodies, dtype=torch.float32) + 0.1
        )  # Positive values

        # Create random inverse inertia tensors (symmetric positive-definite)
        body_inv_inertia = torch.zeros((num_bodies, 3, 3), dtype=torch.float32)
        for i in range(num_bodies):
            A = torch.rand((3, 3), dtype=torch.float32)
            body_inv_inertia[i] = (
                A @ A.T + torch.eye(3) * 0.1
            )  # Ensure positive-definite

        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32)
        dt = 0.01

        # Compute results using original function (body-by-body)
        original_body_q = torch.zeros_like(body_q)
        original_body_qd = torch.zeros_like(body_qd)

        for i in range(num_bodies):
            original_body_q[i], original_body_qd[i] = integrate_body(
                body_q[i],
                body_qd[i],
                body_f[i],
                body_inv_mass[i],
                body_inv_inertia[i],
                gravity,
                dt,
            )

        # Compute results using vectorized function
        vectorized_body_q, vectorized_body_qd = integrate_bodies_vectorized(
            body_q, body_qd, body_f, body_inv_mass, body_inv_inertia, gravity, dt
        )

        # Compare results
        q_error = torch.abs(original_body_q - vectorized_body_q).max().item()
        qd_error = torch.abs(original_body_qd - vectorized_body_qd).max().item()

        print(f"  Max position error: {q_error:.8f}")
        print(f"  Max velocity error: {qd_error:.8f}")

        # Test if results are within acceptable tolerance
        tolerance = 1e-5
        if q_error < tolerance and qd_error < tolerance:
            print(f"  ✓ Results match within tolerance {tolerance}")
        else:
            print(f"  ✗ Results differ beyond tolerance {tolerance}")

            # Print detailed errors for debugging
            print("\nDetailed error analysis:")
            for i in range(
                min(num_bodies, 2)
            ):  # Only print first two bodies in case of many
                print(f"\nBody {i}:")
                print(f"  Original q:  {original_body_q[i]}")
                print(f"  Vectorized q: {vectorized_body_q[i]}")
                print(f"  Q diff:       {original_body_q[i] - vectorized_body_q[i]}")
                print(f"  Original qd:  {original_body_qd[i]}")
                print(f"  Vectorized qd: {vectorized_body_qd[i]}")
                print(f"  QD diff:      {original_body_qd[i] - vectorized_body_qd[i]}")


def test_integration_performance():
    """Compare performance between original and vectorized integration."""

    print("\n\nPerformance comparison:")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test with increasing numbers of bodies to see scaling
    for num_bodies in [10, 100, 1000]:
        print(f"\nTesting with {num_bodies} bodies")

        # Create random test data
        body_q = torch.rand((num_bodies, 7), dtype=torch.float32)
        # Normalize quaternions in the input
        for i in range(num_bodies):
            body_q[i, 3:] = normalize_quat(body_q[i, 3:])

        body_qd = torch.rand((num_bodies, 6), dtype=torch.float32) * 2 - 1
        body_f = torch.rand((num_bodies, 6), dtype=torch.float32) * 10 - 5
        body_inv_mass = torch.rand(num_bodies, dtype=torch.float32) + 0.1

        body_inv_inertia = torch.zeros((num_bodies, 3, 3), dtype=torch.float32)
        for i in range(num_bodies):
            A = torch.rand((3, 3), dtype=torch.float32)
            body_inv_inertia[i] = A @ A.T + torch.eye(3) * 0.1

        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32)
        dt = 0.01

        # Time the original function
        start_time = time.time()
        for _ in range(10):  # Run multiple times to get more stable measurements
            original_body_q = torch.zeros_like(body_q)
            original_body_qd = torch.zeros_like(body_qd)

            for i in range(num_bodies):
                original_body_q[i], original_body_qd[i] = integrate_body(
                    body_q[i],
                    body_qd[i],
                    body_f[i],
                    body_inv_mass[i],
                    body_inv_inertia[i],
                    gravity,
                    dt,
                )

        original_time = (time.time() - start_time) / 10

        # Time the vectorized function
        start_time = time.time()
        for _ in range(10):  # Run multiple times to get more stable measurements
            vectorized_body_q, vectorized_body_qd = integrate_bodies_vectorized(
                body_q, body_qd, body_f, body_inv_mass, body_inv_inertia, gravity, dt
            )

        vectorized_time = (time.time() - start_time) / 10

        # Calculate speedup
        speedup = original_time / vectorized_time

        print(f"  Original function: {original_time:.6f} seconds")
        print(f"  Vectorized function: {vectorized_time:.6f} seconds")
        print(f"  Speedup: {speedup:.2f}x")


def main():
    test_integration_equivalence()
    test_integration_performance()


if __name__ == "__main__":
    main()
