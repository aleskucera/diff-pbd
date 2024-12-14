import numpy as np
import pytest
import torch
from pbd_torch.constants import *
from pbd_torch.correction import *
from pbd_torch.transform import *

# Set random seed for reproducibility
torch.manual_seed(0)


@pytest.mark.parametrize(
    "u_a, u_b, q_a, q_b, I_a_inv, I_b_inv",
    [
        # Case 1: No rotation needed
        (
            torch.tensor([1.0, 0.0, 0.0]),  # u_a
            torch.tensor([1.0, 0.0, 0.0]),  # u_b
            ROT_IDENTITY,  # q_a
            ROT_IDENTITY,  # q_b
            torch.eye(3),  # I_a_inv
            torch.eye(3)),  # I_b_inv

        # Case: 2: 90 degree rotation needed
        (
            torch.tensor([1.0, 0.0, 0.0]),  # u_a
            torch.tensor([1.0, 0.0, 0.0]),  # u_b
            ROT_NEG_90_X,  # q_a
            ROT_IDENTITY,  # q_b
            torch.eye(3),  # I_a_inv
            torch.eye(3)),  # I_b_inv

        # Case 3: 90 degree rotation needed and the body A has infinite inertia
        (
            torch.tensor([1.0, 0.0, 0.0]),  # u_a
            torch.tensor([1.0, 0.0, 0.0]),  # u_b
            ROT_NEG_90_X,  # q_a
            ROT_IDENTITY,  # q_b
            torch.zeros((3, 3)),  # I_a_inv
            torch.eye(3)),  # I_b_inv

        # Case 4: 90 degree rotation needed and the body B has infinite inertia
        (
            torch.tensor([1.0, 0.0, 0.0]),  # u_a
            torch.tensor([1.0, 0.0, 0.0]),  # u_b
            ROT_NEG_90_X,  # q_a
            ROT_IDENTITY,  # q_b
            torch.eye(3),  # I_a_inv
            torch.zeros((3, 3))),  # I_b_inv
    ])
def test_angular_correction(u_a: torch.Tensor, u_b: torch.Tensor,
                            q_a: torch.Tensor, q_b: torch.Tensor,
                            I_a_inv: torch.Tensor, I_b_inv: torch.Tensor):
    n_steps = 5

    u_a = u_a / torch.linalg.norm(u_a)
    u_b = u_b / torch.linalg.norm(u_b)

    for _ in range(n_steps):
        dq_a, dq_b = angular_delta(u_a, u_b, q_a, q_b, I_a_inv, I_b_inv)

        q_a += dq_a
        q_b += dq_b
        q_a = normalize_quat(q_a)
        q_b = normalize_quat(q_b)

    u_a_w = rotate_vectors(u_a, q_a)
    u_b_w = rotate_vectors(u_b, q_b)
    delta_rot = torch.linalg.cross(u_a_w, u_b_w)

    assert torch.allclose(delta_rot, torch.tensor([0.0, 0.0, 0.0]), atol=1e-4)


def test_fixed_random_cases():
    n_cases = 10

    for _ in range(n_cases):
        u_a = torch.rand(3)
        u_b = torch.rand(3)
        q_a = torch.rand(4)
        q_b = torch.rand(4)
        q_a = normalize_quat(q_a)
        q_b = normalize_quat(q_b)
        I_a_inv = torch.eye(3)
        I_b_inv = torch.eye(3)

        test_angular_correction(u_a, u_b, q_a, q_b, I_a_inv, I_b_inv)
