import numpy as np
import cupy as cp
from cupy.typing import DTypeLike

from .activation import ActivationFunction
from src.core import Tensor, Config


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
    __slots__ = "ALPHA", "_dtype"

    ALPHA: float

    _dtype: DTypeLike

    def __init__(self, *, alpha: float = 0.01, dtype: DTypeLike = None) -> None:
        """
        Initialize LeakyRelu with given alpha parameter.
        Args:
            alpha: Slope for negative values. Must be positive.
        """
        if alpha <= 0:
            raise ValueError(f"Alpha must be positive. Got {alpha}")
        
        self._dtype = np.dtype(dtype) if dtype is not None else Config.default_dtype
        
        self.ALPHA = alpha

        if not np.issubdtype(self._dtype, np.floating):
            raise TypeError("Alpha must be a float and dtype or default dtype is not a floating type."
                            f"Got {self._dtype}, use ReLU instead or floats.")


    def __call__[T](self, arr: Tensor[T]) -> Tensor[T]:
        xp = cp.get_array_module(arr.data)
        return arr * xp.where(arr > 0, xp.ones((1,), dtype=self._dtype), xp.array([self.ALPHA], dtype=self._dtype))
