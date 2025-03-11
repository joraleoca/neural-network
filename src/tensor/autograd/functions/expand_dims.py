import cupy as cp

from ..function import Function
from ... import tensor as t
from ..context import Context


class ExpandDims(Function):
    """Expand dims function."""

    @staticmethod
    def forward(a: "t.Tensor", axis: int | list[int] | tuple[int, ...], *, inplace: bool = False) -> "t.Tensor":
        if inplace:
            raise NotImplementedError("Inplace expand dims is not supported.")

        xp = cp.get_array_module(a.data)

        out = t.Tensor(
            xp.expand_dims(a.data, axis),
            requires_grad=Function._requires_grad(a),
            device=Function._select_device(a),
        )

        if out.requires_grad:
            ctx = Context(a, result=out, backward_fn=ExpandDims.backward)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        a = ctx.args[0]

        if a.requires_grad:
            grad = ctx.result.grad
            xp = cp.get_array_module(grad)

            a.update_grad(xp.reshape(grad, a.shape))
