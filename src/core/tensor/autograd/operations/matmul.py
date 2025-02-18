import cupy as cp

from ..function import Function
from ... import tensor
from src.core.tensor.autograd.utils import update_tensor_grad


class Matmul(Function):
    """Function that computes the matrix multiplication of two tensors."""

    def __init__(self, a: "tensor.Tensor", b: "tensor.Tensor") -> None:
        self.args = (a, b)

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        a, b = self.args

        if inplace:
            a.data[:] @= b.data
            return a

        self.result = tensor.Tensor(
            a.data @ b.data,
            requires_grad=a.requires_grad or b.requires_grad,
        )

        return self.result

    def backward(self) -> None:
        a, b = self.args
        grad = self.result.grad

        grad = cp.get_array_module(grad).atleast_2d(grad)

        if a.requires_grad:
            gr = grad @ cp.get_array_module(b.data).atleast_2d(b.data).T

            update_tensor_grad(a, gr)

        if b.requires_grad:
            gr = cp.get_array_module(a.data).atleast_2d(a.data).T @ grad

            update_tensor_grad(b, gr)