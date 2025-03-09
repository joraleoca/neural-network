from typing import ClassVar

from .activation import ActivationFunction
from src.tensor import Tensor, T, op


class Sigmoid(ActivationFunction):
    """
    Sigmoid (logistic) activation function.

    Computes the function: f(x) = 1 / (1 + exp(-x))
    """

    required_fields: ClassVar[tuple[str, ...]] = ()

    def __call__(self, arr: Tensor[T]) -> Tensor[T]:
        return 1 / (1 + op.exp(-arr))

    def data_to_store(self) -> dict[str, None]:
        return {}
    
    @staticmethod
    def from_data(data: dict[str, None] | None = None) -> "Sigmoid":
        return Sigmoid()
