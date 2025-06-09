import cupy as cp
from numpy.typing import DTypeLike

from ..function import Function
from ... import tensor as t
from ..context import Context


class Concat(Function):
    """Concat function."""

    @staticmethod
    def forward(*args: "t.Tensor", axis: int = 0, dtype: str | DTypeLike = None, inplace: bool = False) -> "t.Tensor":
        if inplace:
            raise NotImplementedError("Inplace concat is not supported.")

        xp = cp.get_array_module(*args)
        out = t.Tensor(
            xp.concat(args, axis=axis, dtype=dtype),
            requires_grad=Function._requires_grad(*args),
            device=Function._select_device(*args),
        )

        if out.requires_grad:
            ctx = Context(*args, result=out, backward_fn=Concat.backward, axis=axis)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        grad = ctx.result.grad
        axis = ctx.kwargs["axis"]

        xp = cp.get_array_module(*ctx.args)
        sizes = [arg.shape[axis] for arg in ctx.args]
        indices = xp.cumsum(sizes[:-1])
        grads = xp.split(grad, indices, axis)

        for arg, g in zip(ctx.args, grads):
            arg.update_grad(g)
