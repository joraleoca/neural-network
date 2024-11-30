from .activation import ActivationFunction
from core import Tensor


class Relu(ActivationFunction):
    """
    Rectified Linear Unit (ReLU) activation function.

    Computes the function: f(x) = max(0, x)
    """

    def __call__(self, arr: Tensor) -> Tensor:
        return arr * (arr > 0)
