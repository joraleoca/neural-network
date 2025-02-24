from copy import deepcopy

import cupy as cp

from ..function import Function
from ... import tensor


class Argmax(Function):
    """Function that computes the indices of the maximum values along an axis."""

    __slots__ = "axis", "keepdims"

    def __init__(self, a: "tensor.Tensor", *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
        self.args = (a,)
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        a = self.args[0]

        if inplace:
            raise ValueError("Inplace argmax is not supported.")

        xp = cp.get_array_module(a.data)
        self.result = tensor.Tensor(
            xp.argmax(a.data, axis=self.axis, keepdims=self.keepdims),
            requires_grad=a.requires_grad,
        )

        return self.result

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            # Not differentiable, but we need to propagate the gradient
            if a.grad is None:
                a.grad = deepcopy(grad)
            else:
                a.grad += grad
