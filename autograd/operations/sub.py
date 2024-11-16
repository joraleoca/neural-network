from numpy.typing import NDArray

from .function import Function

from ..context import Context
import tensor


class Sub(Function):
    @staticmethod
    def forward(
        ctx: Context, a: "tensor.Tensor", b: "tensor.Tensor"
    ) -> "tensor.Tensor":
        ctx.data = (a, b)

        result_requires_grad = a.requires_grad or b.requires_grad

        if result_requires_grad:
            ctx.backwards_func = lambda self, grad: Sub.backward(self, grad)

        return tensor.Tensor(
            a.data - b.data,
            dtype=a.dtype,
            requires_grad=result_requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: NDArray) -> None:
        a, b = ctx.data

        if a.requires_grad:
            a.grad += grad
        if b.requires_grad:
            b.grad -= grad
