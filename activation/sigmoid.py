import numpy as np
from numpy.typing import NDArray

from .activation import FunctionActivation


class Sigmoid(FunctionActivation):
    """
    Sigmoid (logistic) activation function.

    Computes the function: f(x) = 1 / (1 + exp(-x))

    The derivative is: f'(x) = f(x)(1 - f(x))
    """

    def activate(self, data: NDArray) -> NDArray:
        self._validate_input(data)
        return 1 / (1 + np.exp(-data))

    def derivative(self, data: NDArray) -> NDArray:
        self._validate_input(data)
        sigmoid = self.activate(data)
        return sigmoid * (1 - sigmoid)
