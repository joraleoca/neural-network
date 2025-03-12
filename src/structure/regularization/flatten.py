from typing import Any, ClassVar

import numpy as np
from numpy.random import Generator

from ..layer import Layer
from src.tensor import Tensor, T


class Flatten(Layer):
    """Flatten layer."""

    __slots__ = "rng"

    rng: Generator

    required_fields: ClassVar[tuple[str, ...]] = tuple()

    def __init__(self, rng: Any = None) -> None:
        """
        Initialize the layer.

        Args:
            rng: Random number generator
        """
        self.rng = np.random.default_rng(rng)

    def __call__(self, data: Tensor[T]) -> Tensor[T]:
        return data.reshape((data.shape[0], -1), inplace=False)

    def data_to_store(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def from_data(data: dict[str, Any] | None = None) -> "Flatten":
        return Flatten()
