from ... import tensor as t
from ..function import Function
from ..context import Context


class Sub(Function):
    """Function that computes the element-wise subtraction of two tensors."""

    @staticmethod
    def forward(a: "t.Tensor", b: "t.Tensor", *, inplace: bool = False) -> "t.Tensor":
        if inplace:
            a.data -= b.data
            return a

        out = t.Tensor(
            a.data - b.data,
            requires_grad=Function._requires_grad(a, b),
            device=Function._select_device(a, b),
        )

        if out.requires_grad:
            ctx = Context(a, b, result=out, backward_fn=Sub.backward)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        a, b = ctx.args
        grad = ctx.result.grad

        if a.requires_grad:
            a.update_grad(grad)

        if b.requires_grad:
            b.update_grad(-grad)
