from typing import Callable

from .. import tensor as t


class Context:
    __slots__ = "args", "result", "_backward_fn", "kwargs"

    args: tuple["t.Tensor", ...]
    result: "t.Tensor"
    _backward_fn: Callable[["Context"], None]
    kwargs: dict | None

    def __init__(
        self, *args: "t.Tensor", result: "t.Tensor", backward_fn: Callable[["Context"], None], **kwargs
    ) -> None:
        """
        Initializes the context for the autograd operations.

        Args:
            *args (t.Tensor): The input tensors.
            result (t.Tensor): The output tensor.
            _backward_fn (Callable[[], None]): The function to be called for the backward pass.
            **kwargs: Additional necessary arguments.
        """
        self.args = args
        self.result = result
        self._backward_fn = backward_fn
        self.kwargs = kwargs

    def backward(self) -> None:
        """
        Computes the gradient of the loss with respect to the input tensors.
        """
        if self.result.grad is None:
            raise ValueError("Cannot compute gradient of a tensor that does not require grad.")

        return self._backward_fn(self)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Context) and id(self) == id(other)
