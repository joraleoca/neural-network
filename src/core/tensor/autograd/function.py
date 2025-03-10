from abc import ABC, abstractmethod

from numpy.typing import NDArray

from ... import config
from .. import tensor as t


class Function(ABC):
    """
    Abstract base class for defining autograd operations.
    """

    __slots__ = "args", "result"

    args: "tuple[t.Tensor, ...]"
    result: "t.Tensor"

    @abstractmethod
    def __call__(
        self,
        *,
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

    def _create_output_tensor(self, data: NDArray) -> "t.Tensor":
        """
        Creates a tensor as result with the given data.\n
        It stores the tensor in the result attribute if the operation requires grad.

        Args:
            data (NDArray): The data to be stored in the tensor.

        Returns:
            Tensor: The tensor with the given data.
        """
        def_device = config.Config.default_device
        
        for arg in self.args:
            if arg.device == def_device:
                device = def_device
                break
            else:
                device = arg.device

        required_grad = any(arg.requires_grad for arg in self.args)

        out = t.Tensor(
            data,
            requires_grad=required_grad,
            device=device,
        )

        if required_grad:
            self.result = out

        return out
