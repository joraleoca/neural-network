import cupy as cp
from src.constants import EPSILON

from ..function import Function
from ... import tensor


class CategoricalCrossentropy(Function):
    """
    Categorical Crossentropy loss.

    This funtion is used because the softmax derivative requires the jacobian matrix and is computationally expensive.
    Also, the autograd does not support jacobian matrix.
    """

    def __init__(self, logits: "tensor.Tensor", expected: "tensor.Tensor"):
        self.args = (logits, expected)

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        predicted, expected = self.args
        xp = cp.get_array_module(*self.args)

        predicted.data = predicted.data.clip(EPSILON, 1 - EPSILON)
        expected.data = expected.data.clip(EPSILON, 1 - EPSILON)

        data = -xp.sum(expected.data * xp.log(predicted.data), axis=-1)

        if inplace:
            predicted.data = data
            return predicted

        return self._create_output_tensor(data)

    def backward(self) -> None:
        predicted, expected = self.args
        grad = self.result.grad.reshape(-1, 1)

        if predicted.requires_grad:
            gr = (
                grad
                * (predicted.data - expected.data)
                / (predicted.data * (1 - predicted.data) + EPSILON)
            )

            if predicted.grad is None:
                predicted.grad = gr
            else:
                predicted.grad += gr

        if expected.requires_grad:
            raise NotImplementedError(
                "Backward pass for expected tensor not implemented."
            )
