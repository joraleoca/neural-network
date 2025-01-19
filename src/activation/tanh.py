from .activation import ActivationFunction
from src.core import Tensor
from ..core.tensor import op


class Tanh(ActivationFunction):
    """
    Hyperbolic tangent activation function.

    Computes the function: f(x) = tanh(x)
    """

    def __call__[T](self, arr: Tensor[T]) -> Tensor[T]:
        return op.tanh(arr)
