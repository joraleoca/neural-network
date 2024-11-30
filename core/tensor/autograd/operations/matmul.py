import numpy as np

from ..function import Function
from ... import tensor


class Matmul(Function):
    def __init__(self, a: "tensor.Tensor", b: "tensor.Tensor") -> None:
        self.args = (a, b)

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        a, b = self.args

        if inplace:
            a.data[:] @= b.data
            return a

        self.result = tensor.Tensor(
            a.data @ b.data,
            dtype=a.dtype if a.dtype == b.dtype else np.floating,
            requires_grad=a.requires_grad or b.requires_grad,
        )

        return self.result

    def backward(self) -> None:
        a, b = self.args
        grad = self.result.grad

        if a.requires_grad:
            a.grad += grad @ b.data.T
        if b.requires_grad:
            b.grad += a.data.T @ grad
