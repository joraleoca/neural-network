from typing import ClassVar

from .activation import ActivationFunction
from src.core import Tensor, T, op


class Tanh(ActivationFunction):
    """
    Hyperbolic tangent activation function.

    Computes the function: f(x) = tanh(x)
    """

    required_fields: ClassVar[tuple[str, ...]] = ()

    def __call__(self, arr: Tensor[T]) -> Tensor[T]:
        return op.tanh(arr)

    def data_to_store(self) -> dict[str, None]:
        return {}
    
    @staticmethod
    def from_data(data: dict[str, None] | None = None) -> "Tanh":
        return Tanh() 