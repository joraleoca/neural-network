import numpy as np
from numpy.typing import NDArray

from .function import Function

from ..context import Context
import tensor


class Pow(Function):
    @staticmethod
    def forward(
        ctx: Context, a: "tensor.Tensor", b: "tensor.Tensor", *, inplace: bool = False
    ) -> "tensor.Tensor":
        ctx.data = (a, b)

        result_requires_grad = a.requires_grad or b.requires_grad

        if result_requires_grad:
            ctx.backwards_func = lambda self, grad: Pow.backward(self, grad)

        if inplace:
            a.data[:] **= b.data
            return a

        return tensor.Tensor(
            a.data**b.data,
            dtype=a.dtype,
            requires_grad=result_requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: NDArray) -> None:
        a, b = ctx.data

        if a.requires_grad:
            a.grad += grad * (b.data * a.data ** (b.data - 1))
        if b.requires_grad:
            if all(a.data > 0):
                b.grad += grad * (a.data**b.data * np.log(a.data))
            else:
                print(
                    f"Warning: Cannot compute gradient with respect to b for a={a.data} <= 0. Gradient not modified."
                )
