from .activation import ActivationFunction
from src.tensor import Tensor, T, op


class Relu(ActivationFunction):
    """
    Rectified Linear Unit (ReLU) activation function.

    Computes the function: f(x) = max(0, x)
    """

    def __call__(self, arr: Tensor[T]) -> Tensor[T]:
        return op.relu(arr)
