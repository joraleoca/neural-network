import numpy as np
from numpy.typing import NDArray

from .activation import ActivationFunction


class Tanh(ActivationFunction):
    """
    Hyperbolic tangent activation function.

    Computes the function: f(x) = tanh(x)

    The derivative is: f'(x) = 1 - tanh²(x)
    """

    def activate(self, data: NDArray) -> NDArray:
        self._validate_input(data)
        return np.tanh(data)

    def derivative(self, data: NDArray) -> NDArray:
        self._validate_input(data)
        return 1 - (np.tanh(data) ** 2)
