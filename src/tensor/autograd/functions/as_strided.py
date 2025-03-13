import cupy as cp

from ..function import Function
from ... import tensor as t
from ..context import Context


class As_Strided(Function):
    """Creates a view of a tensor with the specified strides and shape."""

    @staticmethod
    def forward(
        a: "t.Tensor", shape: tuple[int, ...], strides: tuple[int, ...], *, inplace: bool = False
    ) -> "t.Tensor":
        if inplace:
            raise NotImplementedError("Inplace as_strided is not supported. It always returns a view.")

        xp = cp.get_array_module(a.data)

        out = t.Tensor(
            xp.lib.stride_tricks.as_strided(a.data, shape, strides),
            requires_grad=Function._requires_grad(a),
            device=Function._select_device(a),
        )

        if out.requires_grad:
            ctx = Context(a, result=out, backward_fn=As_Strided.backward, shape=shape, strides=strides)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        a = ctx.args[0]
        if not a.requires_grad:
            return

        xp = cp.get_array_module(a.data)
        gr = xp.zeros_like(a.data)
        result = ctx.result
        shape = ctx.kwargs["shape"]

        strides = xp.array(ctx.kwargs["strides"]) // result.data.itemsize
        indices = xp.indices(shape).reshape(len(shape), -1)
        offsets = xp.tensordot(strides, indices, axes=1).astype(int)

        xp.add.at(gr.ravel(), offsets, result.grad.ravel())

        a.update_grad(gr)
