from ... import tensor as t
from ..context import Context
from ..function import Function


class Neg(Function):
    """Function that computes the element-wise negation of a tensor."""

    @staticmethod
    def forward(a: "t.Tensor", *, inplace: bool = False) -> "t.Tensor":
        if inplace:
            a.data *= -1
            return a

        out = t.Tensor(
            -a.data,
            requires_grad=Function._requires_grad(a),
            device=Function._select_device(a),
        )

        if out.requires_grad:
            ctx = Context(a, result=out, backward_fn=Neg.backward)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        a = ctx.args[0]
        if not a.requires_grad:
            return

        grad = ctx.result.grad

        if a.requires_grad:
            a.update_grad(-grad)
