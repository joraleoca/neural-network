from .activation import ActivationFunction
from src.tensor import Tensor, T, op


class Softmax(ActivationFunction):
    """
    Softmax activation function for multi-class classification.

    Computes the function: f(x_i) = exp(x_i) / Σ(exp(x_j))
    where the sum is over all elements j.
    """

    def __call__(self, arr: Tensor[T]) -> Tensor[T]:
        return op.softmax(arr)
