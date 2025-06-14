from .activation import ActivationFunction
from src.tensor import Tensor, T, op


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
        return op.leaky_relu(arr, self.ALPHA)
