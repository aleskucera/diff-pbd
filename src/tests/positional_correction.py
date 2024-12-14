import torch
import pytest

from pbd_torch.constants import *
from pbd_torch.correction import *

# Set random seed for reproducibility
torch.manual_seed(0)


@pytest.mark.parametrize(
    "x_a, x_b, q_a, q_b, r_a, r_b, m_a_inv, m_b_inv, I_a_inv, I_b_inv",
    [
        # Case 1: Both rs are zero (only positional correction)
        (torch.tensor([0.0, -2.0, 0.0]),  # x_a
         torch.tensor([0.0, 0.0, 2.0]),  # x_b
         ROT_IDENTITY,  # q_a
         ROT_IDENTITY,  # q_b
         torch.zeros(3),  # r_a
         torch.zeros(3),  # r_b
         1.0,  # m_a_inv
         1.0,  # m_b_inv
         torch.eye(3),  # I_a_inv
         torch.eye(3)),  # I_b_inv

        # Case 2: Little difference between rs
        (torch.tensor([0.0, -2.0, 0.0]),  # x_a
         torch.tensor([0.0, 0.0, 2.0]),  # x_b
         ROT_IDENTITY,  # q_a
         ROT_IDENTITY,  # q_b
         torch.tensor([1.0, 1.0, 0.0]),  # r_a
         torch.tensor([0.9, 1.1, 0.0]),  # r_b
         1.0,  # m_a_inv
         1.0,  # m_b_inv
         torch.eye(3),  # I_a_inv
         torch.eye(3)),  # I_b_inv

        # Case 3: Large difference between rs
        (torch.tensor([0.0, -2.0, 0.0]),  # x_a
         torch.tensor([0.0, 0.0, 2.0]),  # x_b
         ROT_IDENTITY,  # q_a
         ROT_IDENTITY,  # q_b
         torch.tensor([1.0, 1.0, 0.0]),  # r_a
         torch.tensor([-1.0, -1.0, 0.0]),  # r_b
         1.0,  # m_a_inv
         1.0,  # m_b_inv
         torch.eye(3),  # I_a_inv
         torch.eye(3)),  # I_b_inv

        # Case 4: Mass A weight is infinite
        (torch.tensor([0.0, 0.0, 0.0]),  # x_a
         torch.tensor([0.0, 0.0, 2.0]),  # x_b
         ROT_IDENTITY,  # q_a
         ROT_IDENTITY,  # q_b
         torch.tensor([1.0, 1.0, 0.0]),  # r_a
         torch.tensor([-1.0, -1.0, 0.0]),  # r_b
         0.0,  # m_a_inv
         1.0,  # m_b_inv
         torch.zeros((3, 3)),  # I_a_inv
         torch.eye(3)),  # I_b_inv

        # Case 5: Mass B weight is infinite
        (torch.tensor([0.0, 0.0, 0.0]),  # x_a
         torch.tensor([0.0, 0.0, 2.0]),  # x_b
         ROT_IDENTITY,  # q_a
         ROT_IDENTITY,  # q_b
         torch.tensor([1.0, 1.0, 0.0]),  # r_a
         torch.tensor([-1.0, -1.0, 0.0]),  # r_b
         1.0,  # m_a_inv
         0.0,  # m_b_inv
         torch.eye(3),  # I_a_inv
         torch.zeros((3, 3))),  # I_b_inv
    ]
)
def test_positional_correction(x_a, x_b, q_a, q_b, r_a, r_b, m_a_inv, m_b_inv, I_a_inv, I_b_inv):
    n_steps = 20

    for _ in range(n_steps):
        dx_a, dx_b, dq_a, dq_b, d_lambda = positional_delta(
            x_a=x_a, x_b=x_b,
            q_a=q_a, q_b=q_b,
            r_a=r_a, r_b=r_b,
            m_a_inv=m_a_inv, m_b_inv=m_b_inv,
            I_a_inv=I_a_inv, I_b_inv=I_b_inv
        )

        x_a += dx_a
        x_b += dx_b
        q_a += dq_a
        q_b += dq_b
        q_a /= torch.norm(q_a)
        q_b /= torch.norm(q_b)

    difference = rotate_vectors(r_a, q_a) + x_a - rotate_vectors(r_b, q_b) - x_b
    assert torch.allclose(difference, torch.zeros(3), atol=1e-2)


def test_random_cases():
    n_cases = 10  # Number of random cases
    for _ in range(n_cases):
        x_a = torch.rand(3)
        x_b = torch.rand(3)
        q_a = torch.rand(4)
        q_b = torch.rand(4)
        q_a /= torch.norm(q_a)
        q_b /= torch.norm(q_b)
        r_a = torch.rand(3)
        r_b = torch.rand(3)
        m_a_inv = torch.rand(1).item()
        m_b_inv = torch.rand(1).item()
        I_a_inv = torch.eye(3) * torch.rand(1).item()
        I_b_inv = torch.eye(3) * torch.rand(1).item()

        test_positional_correction(x_a, x_b, q_a, q_b, r_a, r_b, m_a_inv, m_b_inv, I_a_inv, I_b_inv)

