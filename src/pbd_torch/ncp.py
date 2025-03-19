from typing import Tuple
from typing import Union

import torch


class FisherBurmeister:
    """Fisher-Burmeister function for complementarity problems.

    Implements φ(a,b) = a + b - √(a² + b² + ε) to enforce a ≥ 0, b ≥ 0, a·b = 0 smoothly.
    """

    def __init__(self, epsilon: float = 1e-6):
        """Initialize with a small epsilon for numerical stability.

        Args:
            epsilon: Small value to ensure sqrt and derivatives are well-defined.
        """
        self.epsilon = epsilon

    def evaluate(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute φ(a,b) = a + b - √(a² + b² + ε).

        Args:
            a: First tensor (e.g., shape [batch_size, vector_dim])
            b: Second tensor (same shape as a)

        Returns:
            Tensor of shape matching a and b.
        """
        norm = torch.sqrt(a**2 + b**2 + self.epsilon)
        return a + b - norm

    def derivatives(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute partial derivatives ∂φ/∂a and ∂φ/∂b.

        Args:
            a: First tensor
            b: Second tensor

        Returns:
            Tuple (∂φ/∂a, ∂φ/∂b), each matching the shape of a and b.
        """
        norm = torch.sqrt(a**2 + b**2 + self.epsilon)
        da = 1.0 - a / norm
        db = 1.0 - b / norm
        return da, db

    def derivative_a(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute ∂φ/∂a."""
        da, _ = self.derivatives(a, b)
        return da

    def derivative_b(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute ∂φ/∂b."""
        _, db = self.derivatives(a, b)
        return db

    def __call__(
        self, a: torch.Tensor, b: torch.Tensor, return_derivatives: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Evaluate φ(a,b) and optionally its derivatives.

        Args:
            a: First tensor
            b: Second tensor
            return_derivatives: If True, return (value, ∂φ/∂a, ∂φ/∂b)

        Returns:
            Tensor of φ(a,b) or tuple (φ(a,b), ∂φ/∂a, ∂φ/∂b).
        """
        norm = torch.sqrt(a**2 + b**2 + self.epsilon)
        value = a + b - norm
        if return_derivatives:
            da = 1.0 - a / norm
            db = 1.0 - b / norm
            return value, da, db
        return value

