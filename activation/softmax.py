import numpy as np

from .activation import ActivationFunction
from core import Tensor, op


class Softmax(ActivationFunction):
    """
    Softmax activation function for multi-class classification.

    Computes the function: f(x_i) = exp(x_i) / Σ(exp(x_j))
    where the sum is over all elements j.
    """

    def __call__(self, arr: Tensor) -> Tensor:
        exp_shifted = op.exp(arr - np.max(arr.data))
        softmax = exp_shifted / op.sum(exp_shifted)

        return softmax
