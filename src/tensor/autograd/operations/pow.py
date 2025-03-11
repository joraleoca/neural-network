import warnings

import cupy as cp

from src.constants import EPSILON

from ... import tensor as t
from ..context import Context
from ..function import Function


class Pow(Function):
    """Function that computes the element-wise power of two tensors."""

    @staticmethod
    def forward(a: "t.Tensor", b: "t.Tensor", *, inplace: bool = False) -> "t.Tensor":
        if inplace:
            a.data **= b.data
            return a

        out = t.Tensor(
            a.data**b.data,
            requires_grad=Function._requires_grad(a, b),
            device=Function._select_device(a, b),
        )

        if out.requires_grad:
            ctx = Context(a, b, result=out, backward_fn=Pow.backward)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        a, b = ctx.args
        grad = ctx.result.grad

        if a.requires_grad:
            gr = grad * (b.data * a.data ** (b.data - 1))

            a.update_grad(gr)

        if b.requires_grad:
            xp = cp.get_array_module(a.data)
            if not xp.all(a.data > 0):
                warnings.warn(f"Cannot compute gradient with respect to b for a={a.data} <= 0. Gradient not modified.")
                return

            xp = cp.get_array_module(a.data, b.data)
            gr = grad * (a.data**b.data * xp.log(a.data + EPSILON))

            b.update_grad(gr)
