import numpy as np
import cupy as cp

from ..function import Function
from ... import tensor as t
from ..context import Context


class Where(Function):
    """
    Function that selects elements from `x` or `y` based on the condition tensor.
    """

    @staticmethod
    def forward(condition: "t.Tensor[np.bool]", x: "t.Tensor", y: "t.Tensor", *, inplace: bool = False) -> "t.Tensor":
        if inplace:
            raise ValueError("Inplace where is not supported.")

        xp = cp.get_array_module(x.data, y.data)

        out = t.Tensor(
            xp.where(condition.data, x.data, y.data),
            requires_grad=Function._requires_grad(x, y),
            device=Function._select_device(x, y),
        )

        if out.requires_grad:
            ctx = Context(condition, x, y, result=out, backward_fn=Where.backward)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        condition, x, y = ctx.args
        grad = ctx.result.grad

        if x.requires_grad:
            x.update_grad(grad * condition.data)

        if y.requires_grad:
            y.update_grad(grad * ~condition.data)
