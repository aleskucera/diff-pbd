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
    
    
class ScaledFisherBurmeister:
    def __init__(self, alpha: float, beta: float, epsilon: float = 1e-6):
        assert alpha > 0 and beta > 0, "Scaling factors must be positive."
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def evaluate(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        scaled_a = self.alpha * a
        scaled_b = self.beta * b
        norm = torch.sqrt(scaled_a**2 + scaled_b**2 + self.epsilon)
        return scaled_a + scaled_b - norm

    def derivatives(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scaled_a = self.alpha * a
        scaled_b = self.beta * b
        norm = torch.sqrt(scaled_a**2 + scaled_b**2 + self.epsilon)
        da = self.alpha * (1.0 - scaled_a / norm)
        db = self.beta * (1.0 - scaled_b / norm)
        return da, db

    def derivative_a(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        da, _ = self.derivatives(a, b)
        return da

    def derivative_b(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        _, db = self.derivatives(a, b)
        return db

    def __call__(self, a: torch.Tensor, b: torch.Tensor, return_derivatives: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        scaled_a = self.alpha * a
        scaled_b = self.beta * b
        norm = torch.sqrt(scaled_a**2 + scaled_b**2 + self.epsilon)
        value = scaled_a + scaled_b - norm
        if return_derivatives:
            da = self.alpha * (1.0 - scaled_a / norm)
            db = self.beta * (1.0 - scaled_b / norm)
            return value, da, db
        return value
    
    
class PenalizedFisherBurmeister:
    """Penalized Fisher-Burmeister function for complementarity problems.

    Implements φ(a,b) = a + b - √(a² + b² + 2 p a b + ε) to enforce a ≥ 0, b ≥ 0, a·b = 0 smoothly.
    """

    def __init__(self, p: float, epsilon: float = 1e-6):
        """Initialize with penalty parameter p and small epsilon for numerical stability.

        Args:
            p: Penalty parameter, typically > 0.
            epsilon: Small value to ensure sqrt and derivatives are well-defined.
        """
        assert p >= 0, "Penalty parameter p must be non-negative."
        self.p = p
        self.epsilon = epsilon

    def evaluate(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute φ(a,b) = a + b - √(a² + b² + 2 p a b + ε).

        Args:
            a: First tensor (e.g., shape [batch_size, vector_dim])
            b: Second tensor (same shape as a)

        Returns:
            Tensor of shape matching a and b.
        """
        norm = torch.sqrt(a**2 + b**2 + 2 * self.p * a * b + self.epsilon)
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
        norm = torch.sqrt(a**2 + b**2 + 2 * self.p * a * b + self.epsilon)
        da = 1.0 - (a + self.p * b) / norm
        db = 1.0 - (b + self.p * a) / norm
        return da, db

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
        value = self.evaluate(a, b)
        if return_derivatives:
            da, db = self.derivatives(a, b)
            return value, da, db
        return value
    
class GeneralizedFisherBurmeister:
    """Generalized Fisher-Burmeister function for complementarity problems.

    Implements φ(a,b) = a + b - ( |a|^p + |b|^p + ε )^{1/p} to enforce a ≥ 0, b ≥ 0, a·b = 0 smoothly.
    """

    def __init__(self, p: float, epsilon: float = 1e-6):
        """Initialize with norm parameter p and small epsilon for numerical stability.

        Args:
            p: Norm parameter, typically > 1.
            epsilon: Small value to ensure the function and derivatives are well-defined.
        """
        assert p > 1, "Norm parameter p must be greater than 1."
        self.p = p
        self.epsilon = epsilon

    def evaluate(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute φ(a,b) = a + b - ( |a|^p + |b|^p + ε )^{1/p}.

        Args:
            a: First tensor (e.g., shape [batch_size, vector_dim])
            b: Second tensor (same shape as a)

        Returns:
            Tensor of shape matching a and b.
        """
        norm = (torch.abs(a)**self.p + torch.abs(b)**self.p + self.epsilon)**(1.0 / self.p)
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
        s = torch.abs(a)**self.p + torch.abs(b)**self.p + self.epsilon
        norm = s**(1.0 / self.p)
        # Handle p=1 separately as an approximation due to non-differentiability
        if self.p == 1.0:
            da = 1.0 - torch.sign(a) / (norm + self.epsilon)
            db = 1.0 - torch.sign(b) / (norm + self.epsilon)
        else:
            da = 1.0 - (torch.sign(a) * torch.abs(a)**(self.p - 1)) / (s**(1 - 1.0 / self.p) + self.epsilon)
            db = 1.0 - (torch.sign(b) * torch.abs(b)**(self.p - 1)) / (s**(1 - 1.0 / self.p) + self.epsilon)
        return da, db

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
        value = self.evaluate(a, b)
        if return_derivatives:
            da, db = self.derivatives(a, b)
            return value, da, db
        return value

