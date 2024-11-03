import numpy as np
from numpy.typing import NDArray

from .activation import FunctionActivation


class Relu(FunctionActivation):
    """
    Rectified Linear Unit (ReLU) activation function.

    Computes the function: f(x) = max(0, x)

    The derivative is:
        f'(x) = 1 if x > 0
        f'(x) = 0 if x ≤ 0
    """

    def activate(self, data: NDArray) -> NDArray:
        self._validate_input(data)
        return np.maximum(0, data)

    def derivative(self, data: NDArray) -> NDArray:
        self._validate_input(data)
        return np.where(data > 0, 1, 0)
