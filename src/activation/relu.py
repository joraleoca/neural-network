from .activation import ActivationFunction
from src.core import Tensor


class Relu(ActivationFunction):
    """
    Rectified Linear Unit (ReLU) activation function.

    Computes the function: f(x) = max(0, x)
    """

    def __call__[T](self, arr: Tensor[T]) -> Tensor[T]:
        return arr * (arr > 0)
