from ..function import Function
from ... import tensor as t
from ..context import Context


class Reshape(Function):
    """Function that reshapes a tensor."""

    @staticmethod
    def forward(a: "t.Tensor", shape: tuple[int, ...], *, inplace: bool = False) -> "t.Tensor":
        if inplace:
            a.data = a.data.reshape(shape)
            return a

        out = t.Tensor(
            a.data.reshape(shape),
            requires_grad=Function._requires_grad(a),
            device=Function._select_device(a),
        )

        if out.requires_grad:
            ctx = Context(a, result=out, backward_fn=Reshape.backward)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        a = ctx.args[0]
        if not a.requires_grad:
            return

        a.update_grad(ctx.result.grad.reshape(a.shape))
