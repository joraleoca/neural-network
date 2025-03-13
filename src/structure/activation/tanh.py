from .activation import ActivationFunction
from src.tensor import Tensor, T, op


class Tanh(ActivationFunction):
    """
    Hyperbolic tangent activation function.

    Computes the function: f(x) = tanh(x)
    """

    def __call__(self, arr: Tensor[T]) -> Tensor[T]:
        return op.tanh(arr)
