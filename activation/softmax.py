import numpy as np
from numpy.typing import NDArray

from .activation import FunctionActivation


class Softmax(FunctionActivation):
    """
    Softmax activation function for multi-class classification.

    Computes the function: f(x_i) = exp(x_i) / Σ(exp(x_j))
    where the sum is over all elements j.

    Note:
        The derivative is not implemented as it's typically combined
        with cross-entropy loss for better numerical stability.
    """

    def activate(self, data: NDArray) -> NDArray:
        self._validate_input(data)
        exp_shifted = np.exp(data - np.max(data, axis=0, keepdims=True))
        return exp_shifted / np.sum(exp_shifted, axis=0, keepdims=True)

    def derivative(self, data: NDArray) -> NDArray:
        raise NotImplementedError("Softmax derivative is not implemented")
