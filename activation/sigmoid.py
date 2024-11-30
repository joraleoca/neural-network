from .activation import ActivationFunction
from core import Tensor, op


class Sigmoid(ActivationFunction):
    """
    Sigmoid (logistic) activation function.

    Computes the function: f(x) = 1 / (1 + exp(-x))
    """

    def __call__(self, arr: Tensor) -> Tensor:
        return 1 / (1 + op.exp(-arr))
