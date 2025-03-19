from typing import Any

from ..layer import Layer
from src.tensor import Tensor, T, op


class Dropout(Layer):
    """Dropout layer."""

    __slots__ = "p", "rng"

    p: float

    rng: Any

    def __init__(self, p: float = 0.0, rng: Any = None) -> None:
        """
        Initializes the Dropout layer.
        Args:
            p: The dropout rate.
            rng: The random number generator.
        """
        if not (0 <= p <= 1):
            raise ValueError(f"The dropout rate must be between 0 and 1. Got {p}.")

        self.p = p
        self.rng = rng

    def __call__(self, data: Tensor[T]) -> Tensor[T]:
        if not data.requires_grad or self.p == 0:
            return data

        return op.dropout(data, self.p, rng=self.rng)
