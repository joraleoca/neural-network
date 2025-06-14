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
    def forward(
        predicted: "t.Tensor", expected: "t.Tensor", *, ignore_token_id: int | None = None, inplace: bool = False
    ) -> "t.Tensor":
        xp = cp.get_array_module(predicted.data)

        predicted.data = predicted.data.clip(EPSILON, 1 - EPSILON)
        expected.data = expected.data.clip(EPSILON, 1 - EPSILON)

        data = -expected.data * xp.log(predicted.data)
        data = xp.sum(data, axis=-1)

        if inplace:
            if ignore_token_id is not None:
                mask = expected.data.argmax(axis=-1) != ignore_token_id
                data = data * mask
                data = data.sum() / (mask.sum() + EPSILON)

            predicted.data = data
            return predicted

        out = t.Tensor(
            data,
            requires_grad=Function._requires_grad(predicted, expected),
            device=Function._select_device(predicted, expected),
        )

        if out.requires_grad:
            ctx = Context(
                predicted,
                expected,
                result=out,
                backward_fn=CategoricalCrossentropy.backward,
                ignore_token_id=ignore_token_id,
            )
            out._grad_ctx = ctx

        if ignore_token_id is not None:
            mask = expected.data.argmax(axis=-1) != ignore_token_id
            out = out * mask
            out = out.sum() / (mask.sum() + EPSILON)

        return out

    @staticmethod
    def backward(ctx: Context) -> None:
        predicted, expected = ctx.args

        if predicted.requires_grad:
            grad = ctx.result.grad.reshape(-1, 1)

            grad = grad * (predicted.data - expected.data) / (predicted.data * (1 - predicted.data) + EPSILON)

            predicted.update_grad(grad)

        if expected.requires_grad:
            raise NotImplementedError("Backward pass for expected tensor not implemented.")
