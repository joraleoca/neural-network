import cupy as cp

from ..function import Function
from ... import tensor as t
from ..context import Context


class Pad(Function):
    """Pad a tensor."""

    @staticmethod
    def forward[T](a: "t.Tensor[T]", pad_width: int | tuple = 0, value: T = 0, *, inplace: bool = False) -> "t.Tensor":
        if inplace:
            raise NotImplementedError("Inplace padding is not supported.")

        xp = cp.get_array_module(a.data)

        out = t.Tensor(
            xp.pad(a.data, pad_width=pad_width, mode="constant", constant_values=value),
            requires_grad=Function._requires_grad(a),
            device=Function._select_device(a),
        )

        if out.requires_grad:
            ctx = Context(a, result=out, backward_fn=Pad.backward, pad_width=pad_width)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        a = ctx.args[0]

        if a.requires_grad:
            a.update_grad(ctx.result.grad[_index(ctx.kwargs["pad_width"])])


def _index(pad_width: int | tuple) -> tuple[slice, ...]:
    if isinstance(pad_width, int):
        return (slice(pad_width, -pad_width if pad_width > 0 else None),)

    if isinstance(pad_width[0], int):
        return (slice(pad_width[0], -pad_width[1] if pad_width[1] > 0 else None),)

    if isinstance(pad_width[0], tuple):
        return tuple(slice(pad[0], -pad[1] if pad[1] > 0 else None) for pad in pad_width)

    raise ValueError("Invalid pad_width format")
