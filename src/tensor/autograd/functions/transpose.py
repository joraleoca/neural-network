import cupy as cp

from ..function import Function
from ... import tensor as t
from ..context import Context


class Transpose(Function):
    """Function that computes the transpose of a tensor."""

    @staticmethod
    def forward(a: "t.Tensor", axes: "int | tuple[int, ...] | None" = None, *, inplace: bool = False) -> "t.Tensor":
        if inplace:
            a.data.transpose(axes)
            return a

        xp = cp.get_array_module(a.data)

        out = t.Tensor(
            xp.transpose(a.data, axes),
            requires_grad=Function._requires_grad(a),
            device=Function._select_device(a),
        )

        if out.requires_grad:
            ctx = Context(a, result=out, backward_fn=Transpose.backward, axes=axes)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        a = ctx.args[0]
        if not a.requires_grad:
            return

        grad = ctx.result.grad
        axes = ctx.kwargs["axes"]

        grad_axes = None
        if axes is not None:
            grad_axes = [0] * len(axes)

            for i, axis in enumerate(axes):
                grad_axes[axis] = i

        xp = cp.get_array_module(a.data)
        grad = xp.transpose(grad, grad_axes)

        a.update_grad(grad)
