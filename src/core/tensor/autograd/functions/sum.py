import cupy as cp
import numpy as np

from ..function import Function
from ... import tensor


class Sum(Function):
    """Function that computes the sum of a tensor."""

    __slots__ = ["axis", "keepdims"]

    def __init__(
        self, a, *, axis: tuple[int, ...] | int | None = None, keepdims: bool = False
    ):
        self.args = (a,)
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        if inplace:
            raise NotImplementedError("Inplace addition is not supported.")

        a = self.args[0]
        xp = cp.get_array_module(a.data)

        self.result = tensor.Tensor(
            xp.sum(a.data, axis=self.axis, keepdims=self.keepdims),
            requires_grad=a.requires_grad,
        )

        return self.result

    def backward(self) -> None:
        a = self.args[0]
        if not a.requires_grad:
            return

        grad = self.result.grad
        xp = cp.get_array_module(grad)

        # Handle scalar gradient (from global sum)
        if xp.isscalar(grad) or grad.size == 1 or self.keepdims:
            xp = cp.get_array_module(a.data)
            gr = grad * xp.ones_like(a.data)
            if a.grad is None:
                a.grad = gr
            else:
                a.grad += gr
            return

        # Handle axis-specific sums
        grad_shape = list(a.data.shape)
        if self.axis is not None:
            xp = cp.get_array_module(a.data)
            for ax in np.atleast_1d(self.axis):
                grad_shape[ax] = 1

        gr = xp.broadcast_to(xp.reshape(grad, grad_shape), a.shape)

        if a.grad is None:
            a.grad = gr
        else:
            a.grad += gr
