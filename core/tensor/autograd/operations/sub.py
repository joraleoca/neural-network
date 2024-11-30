import numpy as np

from ..function import Function
from ... import tensor


class Sub(Function):
    """Function that computes the element-wise subtraction of two tensors."""

    def __init__(self, a: "tensor.Tensor", b: "tensor.Tensor") -> None:
        self.args = (a, b)

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        a, b = self.args

        if inplace:
            a.data[:] -= b.data
            return a

        self.result = tensor.Tensor(
            a.data - b.data,
            dtype=a.dtype if a.dtype == b.dtype else np.floating,
            requires_grad=a.requires_grad or b.requires_grad,
        )

        return self.result

    def backward(self) -> None:
        a, b = self.args
        grad = self.result.grad

        if a.requires_grad:
            if a.size == 1:
                a.grad = np.full_like(grad, a.grad.item())
            a.grad += grad
        if b.requires_grad:
            if b.size == 1:
                b.grad = np.full_like(grad, b.grad.item())
            b.grad -= grad
