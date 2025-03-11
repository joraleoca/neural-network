import cupy as cp

from ... import tensor as t
from ..context import Context
from ..function import Function


class Abs(Function):
    """Function that computes the element-wise absolute value of a tensor."""

    @staticmethod
    def forward(a: "t.Tensor", *, inplace: bool = False) -> "t.Tensor":
        xp = cp.get_array_module(a.data)

        if inplace:
            xp.abs(a.data, out=a.data)
            return a

        out = t.Tensor(
            xp.abs(a.data),
            requires_grad=Function._requires_grad(a),
            device=Function._select_device(a),
        )

        if out.requires_grad:
            ctx = Context(a, result=out, backward_fn=Abs.backward)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        a = ctx.args[0]
        grad = ctx.result.grad

        if a.requires_grad:
            # -1 for negative, 1 for positive, 0 for 0
            gr = grad * ((a.data > 0) - grad * (a.data < 0)) * (a.data != 0)

            a.update_grad(gr)
