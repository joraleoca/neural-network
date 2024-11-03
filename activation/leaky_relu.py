import numpy as np
from numpy.typing import NDArray

from .activation import FunctionActivation


class LeakyRelu(FunctionActivation):
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
        self.ALPHA = alpha

    def activate(self, data: NDArray) -> NDArray:
        self._validate_input(data)
        return np.where(data > 0, data, self.ALPHA * data)

    def derivative(self, data: NDArray) -> NDArray:
        self._validate_input(data)
        return np.where(data > 0, 1, self.ALPHA)
