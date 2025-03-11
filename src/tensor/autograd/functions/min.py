from typing import SupportsIndex, Sequence

import cupy as cp

from ..function import Function
from ... import tensor as t
from ..context import Context


class Min(Function):
    """Function that computes the minimum value of a tensor."""

    @staticmethod
    def forward(
        a: "t.Tensor",
        axis: SupportsIndex | Sequence[SupportsIndex] | None = None,
        keepdims: bool = False,
        *,
        inplace: bool = False,
    ) -> "t.Tensor":
        if inplace:
            raise NotImplementedError("Inplace min is not supported.")

        out = t.Tensor(
            a.data.min(axis=axis, keepdims=keepdims),
            requires_grad=Function._requires_grad(a),
            device=Function._select_device(a),
        )

        if out.requires_grad:
            ctx = Context(a, result=out, backward_fn=Min.backward, axis=axis, keepdims=keepdims)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        a = ctx.args[0]
        if not a.requires_grad:
            return

        grad = ctx.result.grad
        mask = a.data == a.data.min(axis=axis, keepdims=True)

        axis = ctx.kwargs["axis"]
        if axis is not None and not ctx.kwargs["keepdims"]:
            xp = cp.get_array_module(grad)
            grad = xp.expand_dims(grad, axis)

        a.update_grad(grad * mask)
