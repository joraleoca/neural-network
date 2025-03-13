from ... import tensor as t
from ..context import Context
from ..function import Function


class Mul(Function):
    """Function that computes the element-wise multiplication of two tensors."""

    @staticmethod
    def forward(a: "t.Tensor", b: "t.Tensor", *, inplace: bool = False) -> "t.Tensor":
        if inplace:
            a.data *= b.data
            return a

        out = t.Tensor(
            a.data * b.data,
            requires_grad=Function._requires_grad(a, b),
            device=Function._select_device(a, b),
        )

        if out.requires_grad:
            ctx = Context(a, b, result=out, backward_fn=Mul.backward)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        a, b = ctx.args
        grad = ctx.result.grad

        if a.requires_grad:
            a.update_grad(grad * b.data)

        if b.requires_grad:
            b.update_grad(grad * a.data)
