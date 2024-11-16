from numpy.typing import NDArray

from .function import Function

from ..context import Context
import tensor


class Abs(Function):
    @staticmethod
    def forward(
        ctx: Context, a: "tensor.Tensor", *, inplace: bool = False
    ) -> "tensor.Tensor":
        ctx.data = (a,)

        result_requires_grad = a.requires_grad

        if result_requires_grad:
            ctx.backwards_func = lambda self, grad: Abs.backward(self, grad)

        if inplace:
            a.data[:] = abs(a.data)
            return a

        return tensor.Tensor(
            abs(a.data),
            dtype=a.dtype,
            requires_grad=result_requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: NDArray) -> None:
        a = ctx.data[0]

        if a.requires_grad:
            # -1 for negative, 1 for positive, 0 for 0
            a.grad += grad * ((a.data > 0) - grad * (a.data < 0)) * (a.data != 0)
