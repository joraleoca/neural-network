from abc import ABC, abstractmethod

from .. import tensor as t


class Function(ABC):
    """
    Abstract base class for defining autograd operations.
    """

    __slots__ = "args", "result"

    args: "tuple[t.Tensor, ...]"
    result: "t.Tensor"

    @abstractmethod
    def forward(self, *args, inplace: bool = False, **kwargs) -> "t.Tensor":
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

    @staticmethod
    def _select_device(*args: "t.Tensor") -> str:
        """
        Selects the device to store the result tensor.

        Args:
            *args (t.Tensor): The function operands.

        Returns:
            str: The device to store the result tensor.
        """
        def_device = t.Tensor.default_device

        for arg in args:
            if arg.device == def_device:
                return def_device

        return args[0].device

    @staticmethod
    def _requires_grad(*args: "t.Tensor") -> bool:
        """
        Checks if the result tensor requires grad.

        Args:
            *args (t.Tensor): The function operands.

        Returns:
            bool: True if the result tensor requires grad, False otherwise.
        """
        return t.Tensor.grad and any(arg.requires_grad for arg in args)
