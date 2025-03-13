from copy import copy

import cupy as cp

from ... import tensor as t
from ..context import Context
from ..function import Function


class Index(Function):
    """Function that index a tensor."""

    __slots__ = "idx"

    @staticmethod
    def forward(a: "t.Tensor", idx, *, inplace: bool = False) -> "t.Tensor":
        if inplace:
            raise ValueError("Index cannot be inplace.")

        out = t.Tensor(
            a.data[idx],
            requires_grad=Function._requires_grad(a),
            device=Function._select_device(a),
        )

        if out.requires_grad:
            ctx = Context(a, result=out, backward_fn=Index.backward, idx=idx)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        a = ctx.args[0]
        if not a.requires_grad:
            return

        idx = ctx.kwargs["idx"]
        grad = ctx.result.grad

        if grad.size == 1:
            grad = grad.item()

        if a.grad is None:
            xp = cp.get_array_module(a.data)
            a.grad = xp.zeros_like(a.data)
            a.grad[idx] = copy(grad)
        else:
            a.grad[idx] += grad
