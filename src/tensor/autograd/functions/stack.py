import cupy as cp

from ..function import Function
from ... import tensor as t
from ..context import Context


class Stack(Function):
    """Stack function."""

    @staticmethod
    def forward(*args: "t.Tensor", axis: int = 0, inplace: bool = False) -> "t.Tensor":
        if inplace:
            raise NotImplementedError("Inplace stack is not supported.")

        xp = cp.get_array_module(*args)
        out = t.Tensor(
            xp.stack(args, axis=axis),
            requires_grad=Function._requires_grad(*args),
            device=Function._select_device(*args),
        )

        if out.requires_grad:
            ctx = Context(*args, result=out, backward_fn=Stack.backward, axis=axis)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        grad = ctx.result.grad

        xp = cp.get_array_module(*ctx.args)
        grads = xp.unstack(grad, axis=ctx.kwargs["axis"])

        for tensor, grad in zip(ctx.args, grads):
            if tensor.requires_grad:
                tensor.update_grad(grad)
