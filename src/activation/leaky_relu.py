import cupy as cp

from .activation import ActivationFunction
from src.core import Tensor


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

    ALPHA: float

    def __init__(self, *, alpha: float = 0.01) -> None:
        """
        Initialize LeakyRelu with given alpha parameter.
        Args:
            alpha: Slope for negative values. Must be positive.
        """
        if alpha <= 0:
            raise ValueError(f"Alpha must be positive. Got {alpha}")

        self.ALPHA = alpha

    def __call__[T](self, arr: Tensor[T]) -> Tensor[T]:
        xp = cp.get_array_module(arr.data)
        return arr * xp.where(arr > 0, 1, self.ALPHA)
