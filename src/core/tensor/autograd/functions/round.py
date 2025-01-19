import numpy as np

from ..function import Function
from ... import tensor


class Round(Function):
    """Function that rounds the elements of a tensor to the nearest integer."""

    __slots__ = ["decimals"]

    def __init__(self, a: "tensor.Tensor", *, decimals: int = 0):
        self.args = (a,)
        self.decimals = decimals

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        a = self.args[0]

        if inplace:
            a.data.round(self.decimals)
            return a

        self.result = tensor.Tensor(
            np.round(a.data, self.decimals),
            dtype=a.dtype,
            requires_grad=a.requires_grad,
        )

        return self.result

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            # Not differentiable, but we need to propagate the gradient
            if a.grad is None:
                a.grad = grad
            else:
                a.grad += grad
