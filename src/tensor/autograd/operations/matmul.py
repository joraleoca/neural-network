import cupy as cp

from ... import tensor as t
from ..context import Context
from ..function import Function


class Matmul(Function):
    """Function that computes the matrix multiplication of two tensors."""

    @staticmethod
    def forward(a: "t.Tensor", b: "t.Tensor", *, inplace: bool = False) -> "t.Tensor":
        if inplace:
            a.data @= b.data
            return a

        out = t.Tensor(
            a.data @ b.data,
            requires_grad=Function._requires_grad(a, b),
            device=Function._select_device(a, b),
        )

        if out.requires_grad:
            ctx = Context(a, b, result=out, backward_fn=Matmul.backward)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        a, b = ctx.args
        grad = ctx.result.grad

        grad = cp.get_array_module(grad).atleast_2d(grad)

        if a.requires_grad:
            gr = grad @ cp.get_array_module(b.data).atleast_2d(b.data).swapaxes(-1, -2)

            a.update_grad(gr)

        if b.requires_grad:
            gr = cp.get_array_module(a.data).atleast_2d(a.data).swapaxes(-1, -2) @ grad

            b.update_grad(gr)
