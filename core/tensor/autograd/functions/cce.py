import numpy as np

from ..function import Function
from ... import tensor
import core.constants as c


class CategoricalCrossentropy(Function):
    """
    Categorical Crossentropy loss.

    This funtion is used because the softmax derivative requires the jacobian matrix and is computationally expensive.
    Also the autograd does not support jacobian matrix.
    """

    def __init__(self, logits: "tensor.Tensor", expected: "tensor.Tensor"):
        self.args = (logits, expected)

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        predicted, expected = self.args

        if inplace:
            predicted.data = -np.sum(expected.data * np.log(predicted + c.EPSILON))
            return predicted

        self.result = tensor.Tensor(
            -np.sum(expected.data * np.log(predicted + c.EPSILON)),
            dtype=predicted.dtype,
            requires_grad=predicted.requires_grad,
        )

        return self.result

    def backward(self) -> None:
        predicted, expected = self.args
        grad = self.result.grad

        if predicted.requires_grad:
            predicted.grad += (
                grad
                * (predicted - expected.data)
                / (predicted * (1 - predicted) + c.EPSILON)
            )

        if expected.requires_grad:
            raise NotImplementedError(
                "Backward pass for expected tensor not implemented."
            )
