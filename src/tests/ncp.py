import torch
import pytest
from typing import Tuple, Union
from pbd_torch.ncp import FisherBurmeister


class TestFisherBurmeister:
    @pytest.fixture
    def setup_fb(self):
        fb = FisherBurmeister(epsilon=1e-6)
        return fb

    @pytest.mark.parametrize("shape", [
        (1,), (5,), (2, 3), (2, 3, 4),
    ])
    def test_evaluate_shape(self, setup_fb, shape):
        fb = setup_fb
        a = torch.randn(shape)
        b = torch.randn(shape)
        result = fb.evaluate(a, b)
        assert result.shape == shape, f"Expected shape {shape}, got {result.shape}"

    def test_derivative_a(self, setup_fb):
        fb = setup_fb
        a = torch.randn(2, 3, requires_grad=True)
        b = torch.randn(2, 3)

        # Analytical derivative
        analytical_da = fb.derivative_a(a, b)

        # Numerical derivative
        phi = fb.evaluate(a, b)
        numerical_da = torch.zeros_like(a)
        for i in range(a.numel()):
            grad_output = torch.zeros_like(phi)
            grad_output.flatten()[i] = 1.0
            # Retain graph for all but the last iteration
            retain = i < a.numel() - 1
            grad = torch.autograd.grad(phi, a, grad_outputs=grad_output, retain_graph=retain)[0]
            numerical_da.flatten()[i] = grad.flatten()[i]

        assert analytical_da.shape == a.shape
        assert torch.allclose(analytical_da, numerical_da, atol=1e-6), "∂φ/∂a mismatch"

    def test_derivative_b(self, setup_fb):
        fb = setup_fb
        a = torch.randn(2, 3)
        b = torch.randn(2, 3, requires_grad=True)

        # Analytical derivative
        analytical_db = fb.derivative_b(a, b)

        # Numerical derivative
        phi = fb.evaluate(a, b)
        numerical_db = torch.zeros_like(b)
        for i in range(b.numel()):
            grad_output = torch.zeros_like(phi)
            grad_output.flatten()[i] = 1.0
            # Retain graph for all but the last iteration
            retain = i < b.numel() - 1
            grad = torch.autograd.grad(phi, b, grad_outputs=grad_output, retain_graph=retain)[0]
            numerical_db.flatten()[i] = grad.flatten()[i]

        assert analytical_db.shape == b.shape
        assert torch.allclose(analytical_db, numerical_db, atol=1e-6), "∂φ/∂b mismatch"

    def test_derivatives(self, setup_fb):
        fb = setup_fb
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)

        da, db = fb.derivatives(a, b)
        da_individual = fb.derivative_a(a, b)
        db_individual = fb.derivative_b(a, b)

        assert da.shape == a.shape
        assert db.shape == b.shape
        assert torch.allclose(da, da_individual, atol=1e-6), "da from derivatives mismatches derivative_a"
        assert torch.allclose(db, db_individual, atol=1e-6), "db from derivatives mismatches derivative_b"

    @pytest.mark.parametrize("a_val, b_val", [
        (0.0, 0.0),
        (1e-6, 0.0),
        (-1e-6, 0.0),
        (1.0, 1e-6),
    ])
    def test_derivatives_edge_cases(self, setup_fb, a_val, b_val):
        fb = setup_fb
        a = torch.tensor(a_val, requires_grad=True)
        b = torch.tensor(b_val, requires_grad=True)

        # Analytical derivatives
        da, db = fb.derivatives(a, b)

        # Numerical derivatives
        phi = fb.evaluate(a, b)
        numerical_da = torch.autograd.grad(phi, a, retain_graph=True)[0]
        numerical_db = torch.autograd.grad(phi, b, create_graph=False)[0]

        assert torch.allclose(da, numerical_da, atol=1e-6), f"∂φ/∂a mismatch at a={a_val}, b={b_val}"
        assert torch.allclose(db, numerical_db, atol=1e-6), f"∂φ/∂b mismatch at a={a_val}, b={b_val}"


if __name__ == "__main__":
    pytest.main([__file__])