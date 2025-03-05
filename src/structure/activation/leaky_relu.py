from typing import ClassVar

import cupy as cp

from .activation import ActivationFunction
from src.core import Tensor, T


class LeakyRelu(ActivationFunction):
    """
    Leaky Rectified Linear Unit activation function.

    Computes the function:
        f(x) = x if x > 0
        f(x) = αx if x ≤ 0
    where α is a small positive constant.

    Args:
        alpha: Slope for negative values. Defaults to 0.01.
    """
    __slots__ = "ALPHA"

    ALPHA: float

    required_fields: ClassVar[tuple[str, ...]] = ("ALPHA",)

    def __init__(self, alpha: float = 0.01) -> None:
        """
        Initialize LeakyRelu with given alpha parameter.
        Args:
            alpha: Slope for negative values. Must be positive.
            dtype: Data type of the input tensor.
        """
        if alpha <= 0:
            raise ValueError(f"Alpha must be positive. Got {alpha}")
                
        self.ALPHA = alpha

    def __call__(self, arr: Tensor[T]) -> Tensor[T]:
        xp = cp.get_array_module(arr.data)
        return arr * xp.where(arr > 0, 
                              xp.ones((1,), dtype=arr.dtype),
                              xp.array([self.ALPHA], dtype=arr.dtype)
                            )

    def data_to_store(self) -> dict[str, float]:
        return {
            "ALPHA": self.ALPHA,
        }
    
    @staticmethod
    def from_data(data: dict[str, float]) -> "LeakyRelu":
        return LeakyRelu(data["ALPHA"])