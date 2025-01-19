import numpy as np

from ..function import Function
from ... import tensor


class Mean(Function):
    """Function that computes the mean value of a tensor."""

    __slots__ = ["axis"]

    def __init__(
        self,
        a: "tensor.Tensor",
        *,
        axis: int | tuple[int, ...] | None = None,
    ):
        self.args = (a,)
        self.axis = axis

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        if inplace:
            raise NotImplementedError("Inplace mean is not supported.")

        a = self.args[0]

        self.result = tensor.Tensor(
            a.data.mean(axis=self.axis),
            dtype=a.dtype,
            requires_grad=a.requires_grad,
        )

        return self.result

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            if self.axis is not None:
                grad = np.expand_dims(grad, self.axis)

            n = (
                a.data.size
                if self.axis is None
                else np.prod([a.data.shape[i] for i in np.atleast_1d(self.axis)])
            )
            gr = grad / n

            if a.grad is None:
                a.grad = gr
            else:
                a.grad += gr
