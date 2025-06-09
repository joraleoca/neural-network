import cupy as cp

from ..function import Function
from ... import tensor as t
from ..context import Context


class Triu(Function):
    """Function that returns a tensor with the elements below the k-th diagonal zeroed."""

    @staticmethod
    def forward(a: "t.Tensor", k: int, *, inplace: bool = False) -> "t.Tensor":
        if inplace:
            raise NotImplementedError("Inplace triu is not supported.")

        xp = cp.get_array_module(a)
        out = t.Tensor(
            xp.triu(a, k),
            requires_grad=Function._requires_grad(a),
            device=Function._select_device(a),
        )

        if out.requires_grad:
            ctx = Context(a, result=out, backward_fn=Triu.backward, k=0)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        a = ctx.args[0]
        if not a.requires_grad:
            return

        grad = ctx.result.grad

        xp = cp.get_array_module(grad)
        a.update_grad(grad * xp.triu(xp.ones_like(grad), k=ctx.kwargs["k"]))
