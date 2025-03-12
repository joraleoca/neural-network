import cupy as cp

from ..function import Function
from ... import tensor as t
from ..context import Context


class Mean(Function):
    """Function that computes the mean value of a tensor."""

    @staticmethod
    def forward(
        a: "t.Tensor", axis: int | tuple[int, ...] | None = None, keepdims: bool = False, *, inplace: bool = False
    ) -> "t.Tensor":
        if inplace:
            raise NotImplementedError("Inplace mean is not supported.")

        out = t.Tensor(
            a.data.mean(axis=axis, keepdims=keepdims),
            requires_grad=Function._requires_grad(a),
            device=Function._select_device(a),
        )

        if out.requires_grad:
            ctx = Context(a, result=out, backward_fn=Mean.backward, axis=axis, keepdims=keepdims)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        a = ctx.args[0]
        if not a.requires_grad:
            return

        grad = ctx.result.grad
        xp = cp.get_array_module(grad)

        n = a.data.size

        if grad.size == 1 or ctx.kwargs["keepdims"]:
            grad = xp.broadcast_to(grad, a.shape)
        else:
            axis_ = ctx.kwargs["axis"]
            if axis_ is not None:
                if not isinstance(axis_, tuple):
                    axis_ = (axis_,)

                grad = xp.expand_dims(grad, axis_)

                n = 1
                for axis in axis_:
                    n *= a.shape[axis]

        a.update_grad(grad / n)
