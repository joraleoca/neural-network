import cupy as cp

from ..function import Function
from ... import tensor as t
from ..context import Context


class Ravel(Function):
    """Ravel a tensor."""

    @staticmethod
    def forward(a: "t.Tensor", *, inplace: bool = False) -> "t.Tensor":
        if inplace:
            raise NotImplementedError("Inplace ravel is not supported.")

        out = t.Tensor(
            a.data.ravel(),
            requires_grad=Function._requires_grad(a),
            device=Function._select_device(a),
        )

        if out.requires_grad:
            ctx = Context(a, result=out, backward_fn=Ravel.backward)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        a = ctx.args[0]

        if a.requires_grad:
            grad = ctx.result.grad
            xp = cp.get_array_module(grad)

            a.update_grad(xp.reshape(grad, a.shape))
