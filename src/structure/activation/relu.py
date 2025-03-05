from typing import ClassVar

from .activation import ActivationFunction
from src.core import Tensor, T


class Relu(ActivationFunction):
    """
    Rectified Linear Unit (ReLU) activation function.

    Computes the function: f(x) = max(0, x)
    """

    required_fields: ClassVar[tuple[str, ...]] = ()

    def __call__(self, arr: Tensor[T]) -> Tensor[T]:
        return arr * (arr > 0)

    def data_to_store(self) -> dict[str, None]:
        return {}
    
    @staticmethod
    def from_data(data: dict[str, None] | None = None) -> "Relu":
        return Relu()
