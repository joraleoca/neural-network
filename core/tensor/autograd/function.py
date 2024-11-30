from abc import ABC, abstractmethod

from .. import tensor as t


class Function(ABC):
    """
    Abstract base class for defining autograd operations.
    """

    __slots__ = ["args", "result"]

    args: "tuple[t.Tensor, ...]"
    result: "t.Tensor"

    @abstractmethod
    def __call__(
        self,
        *args: "t.Tensor",
        inplace: bool = False,
    ) -> "t.Tensor":
        """
        Invokes the function with the given arguments.
        """
        pass

    @abstractmethod
    def backward(self) -> None:
        """
        Computes the gradient of the loss with respect to the input tensors.
        """
        pass
