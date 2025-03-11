from ..function import Function
from ... import tensor as t
from ..context import Context


class Compose(Function):
    """Compose function."""

    @staticmethod
    def forward(*args: "t.Tensor", inplace: bool = False) -> "t.Tensor":
        if inplace:
            raise NotImplementedError("Inplace compose is not supported.")

        shape = args[0].shape
        if not all(shape == t.shape for t in args):
            raise ValueError("All tensors must be of the same shape")

        out = t.Tensor(
            args,
            requires_grad=Function._requires_grad(*args),
            device=Function._select_device(*args),
        )

        if out.requires_grad:
            ctx = Context(*args, result=out, backward_fn=Compose.backward)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        grad = ctx.result.grad

        for i, tensor in enumerate(ctx.args):
            if tensor.requires_grad:
                tensor.update_grad(grad[i])
