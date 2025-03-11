import cupy as cp
from src.constants import EPSILON

from ..function import Function
from ... import tensor as t
from ..context import Context


class CategoricalCrossentropy(Function):
    """
    Categorical Crossentropy loss.

    This funtion is used because the softmax derivative requires the jacobian matrix and is computationally expensive.
    Also, the autograd does not support jacobian matrix.
    """

    @staticmethod
    def forward(predicted: "t.Tensor", expected: "t.Tensor", *, inplace: bool = False) -> "t.Tensor":
        xp = cp.get_array_module(predicted.data)

        predicted.data = predicted.data.clip(EPSILON, 1 - EPSILON)
        expected.data = expected.data.clip(EPSILON, 1 - EPSILON)

        data = -xp.sum(expected.data * xp.log(predicted.data), axis=-1)

        if inplace:
            predicted.data = data
            return predicted

        out = t.Tensor(
            data,
            requires_grad=Function._requires_grad(predicted, expected),
            device=Function._select_device(predicted, expected),
        )

        if out.requires_grad:
            ctx = Context(predicted, expected, result=out, backward_fn=CategoricalCrossentropy.backward)
            out._grad_ctx = ctx

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        predicted, expected = ctx.args
        grad = ctx.result.grad

        if predicted.requires_grad:
            gr = grad * (predicted.data - expected.data) / (predicted.data * (1 - predicted.data) + EPSILON)

            predicted.update_grad(gr)

        if expected.requires_grad:
            raise NotImplementedError("Backward pass for expected tensor not implemented.")
