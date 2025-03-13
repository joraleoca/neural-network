from typing import SupportsIndex

from ..function import Function
from ... import tensor as t
from ..context import Context


class Argmax(Function):
    """Function that computes the indices of the maximum values along an axis."""

    @staticmethod
    def forward(
        a: "t.Tensor", axis: SupportsIndex | None = None, keepdims: bool = False, *, inplace: bool = False
    ) -> "t.Tensor":
        if inplace:
            raise ValueError("Inplace argmax is not supported.")
        if axis is None and keepdims:
            raise ValueError("keepdims can only be set to True if axis is specified. Got axis=None and keepdims=True")

        out = t.Tensor(
            a.data.argmax(axis=axis, keepdims=keepdims),  # type: ignore Already checked for no valid combination
            requires_grad=Function._requires_grad(a),
            device=Function._select_device(a),
        )

        if out.requires_grad:
            ctx = Context(a, result=out, backward_fn=Argmax.backward)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        a = ctx.args[0]

        if a.requires_grad:
            a.update_grad(ctx.result.grad)  # Not differentiable, so just pass the gradient through
