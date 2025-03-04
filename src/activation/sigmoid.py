from .activation import ActivationFunction
from src.core import Tensor, T
from ..core.tensor import op


class Sigmoid(ActivationFunction):
    """
    Sigmoid (logistic) activation function.

    Computes the function: f(x) = 1 / (1 + exp(-x))
    """

    def __call__(self, arr: Tensor[T]) -> Tensor[T]:
        return 1 / (1 + op.exp(-arr))
