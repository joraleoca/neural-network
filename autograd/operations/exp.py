import numpy as np
from numpy.typing import NDArray

from .function import Function

from ..context import Context
import tensor


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a: "tensor.Tensor") -> "tensor.Tensor":
        ctx.data = (a,)

        if a.requires_grad:
            ctx.backwards_func = lambda self, grad: Exp.backward(self, grad)

        return tensor.Tensor(
            np.exp(a.data),
            dtype=a.dtype,
            requires_grad=a.requires_grad,
        )

    @staticmethod
    def backward(ctx: Context, grad: NDArray) -> None:
        a = ctx.data[0]

        if a.requires_grad:
            a.grad += grad * np.exp(a.data)
