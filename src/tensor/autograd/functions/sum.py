import cupy as cp

from ..function import Function
from ... import tensor as t
from ..context import Context


class Sum(Function):
    """Function that computes the sum of a tensor."""

    @staticmethod
    def forward(
        a: "t.Tensor", axis: tuple[int, ...] | int | None = None, keepdims: bool = False, *, inplace: bool = False
    ) -> "t.Tensor":
        if inplace:
            raise NotImplementedError("Inplace addition is not supported.")

        out = t.Tensor(
            a.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=Function._requires_grad(a),
            device=Function._select_device(a),
        )

        if out.requires_grad:
            ctx = Context(a, result=out, backward_fn=Sum.backward, axis=axis, keepdims=keepdims)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        a = ctx.args[0]
        if not a.requires_grad:
            return

        grad = ctx.result.grad
        xp = cp.get_array_module(grad)

        # Handle scalar gradient (from global sum)
        if xp.isscalar(grad) or grad.size == 1 or ctx.kwargs["keepdims"]:
            gr = grad.reshape(a.shape)
        else:
            # Handle axis-specific sums
            grad_shape = list(a.data.shape)
            axis = ctx.kwargs["axis"]
            if axis is not None:
                for ax in xp.atleast_1d(axis):
                    grad_shape[ax] = 1

            gr = xp.broadcast_to(grad.reshape(grad_shape), a.shape)

        a.update_grad(gr)
