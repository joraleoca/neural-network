from .activation import ActivationFunction
from core import Tensor, op


class Tanh(ActivationFunction):
    """
    Hyperbolic tangent activation function.

    Computes the function: f(x) = tanh(x)
    """

    def __call__(self, arr: Tensor) -> Tensor:
        return op.tanh(arr)
