from .activation import ActivationFunction
from src.core import Tensor
from ..core.tensor import op


class Softmax(ActivationFunction):
    """
    Softmax activation function for multi-class classification.

    Computes the function: f(x_i) = exp(x_i) / Σ(exp(x_j))
    where the sum is over all elements j.
    """

    def __call__[T](self, arr: Tensor[T]) -> Tensor[T]:
        exp_shifted = op.exp(arr - op.max(arr))
        softmax = exp_shifted / op.sum(exp_shifted)

        return softmax
