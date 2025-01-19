import numpy as np

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

        self.result = tensor.Tensor(
            np.transpose(a.data, axes=self.axes),
            dtype=a.dtype,
            requires_grad=a.requires_grad,
        )

        return self.result

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            grad_axes = None
            if self.axes is not None:
                grad_axes = [0] * len(self.axes)

                for i, axis in enumerate(self.axes):
                     grad_axes[axis] = i

            grad = np.transpose(grad, grad_axes)

            if a.grad is None:
                a.grad = grad
            else:
                a.grad += grad.T
