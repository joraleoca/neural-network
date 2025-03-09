from typing import ClassVar

from .activation import ActivationFunction
from src.tensor import Tensor, T, op


class Softmax(ActivationFunction):
    """
    Softmax activation function for multi-class classification.

    Computes the function: f(x_i) = exp(x_i) / Σ(exp(x_j))
    where the sum is over all elements j.
    """

    required_fields: ClassVar[tuple[str, ...]] = ()

    def __call__(self, arr: Tensor[T]) -> Tensor[T]:
        exp_shifted = op.exp(arr - op.max(arr, axis=-1, keepdims=True))
        softmax = exp_shifted / op.sum(exp_shifted, axis=-1, keepdims=True)

        return softmax

    def data_to_store(self) -> dict[str, None]:
        return {}
    
    @staticmethod
    def from_data(data: dict[str, None] | None = None) -> "Softmax":
        return Softmax()
