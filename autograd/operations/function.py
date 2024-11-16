from abc import ABC, abstractmethod

from numpy.typing import NDArray

import tensor
from ..context import Context


class Function(ABC):
    @staticmethod
    @abstractmethod
    def forward(ctx: Context, *args: "tensor.Tensor") -> "tensor.Tensor":
        pass

    @staticmethod
    @abstractmethod
    def backward(ctx: Context, grad: NDArray) -> None:
        pass
