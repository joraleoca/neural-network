import cupy as cp

from ..function import Function
from ... import tensor as t
from ..context import Context


class Tanh(Function):
    """Function that computes the element-wise tangent of a tensor."""

    @staticmethod
    def forward(a: "t.Tensor", *, inplace: bool = False) -> "t.Tensor":
        xp = cp.get_array_module(a.data)

        if inplace:
            xp.tanh(a.data, out=a.data)
            return a

        out = t.Tensor(
            xp.tanh(a.data),
            requires_grad=Function._requires_grad(a),
            device=Function._select_device(a),
        )

        if out.requires_grad:
            ctx = Context(a, result=out, backward_fn=Tanh.backward)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        a = ctx.args[0]
        if not a.requires_grad:
            return

        result = ctx.result

        a.update_grad(result.grad * (1 - (result.data**2)))
