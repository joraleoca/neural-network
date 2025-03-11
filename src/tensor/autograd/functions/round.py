from ..function import Function
from ... import tensor as t
from ..context import Context


class Round(Function):
    """Function that rounds the elements of a tensor to the nearest integer."""

    @staticmethod
    def forward(a: "t.Tensor", decimals: int = 0, *, inplace: bool = False) -> "t.Tensor":
        if inplace:
            a.data.round(decimals, out=a.data)
            return a

        out = t.Tensor(
            a.data.round(decimals),
            requires_grad=Function._requires_grad(a),
            device=Function._select_device(a),
        )

        if out.requires_grad:
            ctx = Context(a, result=out, backward_fn=Round.backward)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        a = ctx.args[0]
        if not a.requires_grad:
            return

        # Round is a non-differentiable function, so the gradient is just propagated
        a.update_grad(ctx.result.grad)
