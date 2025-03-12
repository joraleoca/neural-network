from typing import Any, ClassVar

import numpy as np
from numpy.random import Generator

from ..layer import Layer
from src.tensor import Tensor, T


class Dropout(Layer):
    """Dropout layer."""

    __slots__ = "p", "rng"

    p: float

    rng: Generator

    required_fields: ClassVar[tuple[str, ...]] = ("p",)

    def __init__(self, p: float = 0.0, rng: Any = None) -> None:
        """
        Initializes the Dropout layer.
        Args:
            p: The dropout rate.
            rng: The random number generator.
        """
        if 0 > p > 1:
            raise ValueError(f"The dropout rate must be between 0 and 1. Got {p}.")

        self.p = p
        self.rng = np.random.default_rng(rng)

    def __call__(self, data: Tensor[T]) -> Tensor[T]:
        if not data.requires_grad or self.p == 0:
            return data

        if 0 > self.p > 1:
            raise ValueError("The dropout probability must be between 0 and 1.")

        mask = Tensor(self.rng.binomial(1, 1 - self.p, size=data.shape) / (1 - self.p), dtype=data.dtype)

        return data * mask

    def data_to_store(self) -> dict[str, Any]:
        return {
            "p": self.p,
        }

    @staticmethod
    def from_data(data: dict[str, Any]) -> "Dropout":
        return Dropout(data["p"].item())
