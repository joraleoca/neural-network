from copy import deepcopy

import cupy as cp

from ..function import Function
from ... import tensor


class Transpose(Function):
    """Function that computes the transpose of a tensor."""

    __slots__ = "axes"

    def __init__(self, a: "tensor.Tensor", *, axes: list[int] | tuple[int, ...] | int | None = None) -> None:
        self.args = (a,)
        self.axes = axes

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        a = self.args[0]

        if inplace:
            a.data.transpose(self.axes)
            return a

        xp = cp.get_array_module(a.data)
        return self._create_output_tensor(xp.transpose(a.data, axes=self.axes))

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            grad_axes = None
            if self.axes is not None:
                grad_axes = [0] * len(self.axes)

                for i, axis in enumerate(self.axes):
                     grad_axes[axis] = i

            xp = cp.get_array_module(a.data)
            grad = xp.transpose(grad, grad_axes)

            if a.grad is None:
                a.grad = deepcopy(grad)
            else:
                a.grad += grad.T
