from typing import SupportsIndex
from copy import deepcopy

from ..function import Function
from ... import tensor


class Argmax(Function):
    """Function that computes the indices of the maximum values along an axis."""

    __slots__ = "axis", "keepdims"

    def __init__(self, a: "tensor.Tensor", *, axis: SupportsIndex | None = None, keepdims: bool = False):
        self.args = (a,)
        self.axis = axis
        self.keepdims = keepdims

        if axis is None and keepdims:
            raise ValueError("keepdims can only be set to True if axis is specified. Got axis=None and keepdims=True")

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        if inplace:
            raise ValueError("Inplace argmax is not supported.")

        a = self.args[0]

        return self._create_output_tensor(
            a.data.argmax(axis=self.axis, keepdims=self.keepdims)  # type: ignore Already checked in the constructor
        )

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            # Not differentiable, but we need to propagate the gradient
            if a.grad is None:
                a.grad = deepcopy(grad)
            else:
                a.grad += grad
