import numpy as np
import cupy as cp

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
            requires_grad=a.requires_grad,
        )

        return self.result

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad
        xp = cp.get_array_module(grad)

        if a.requires_grad:
            n = a.data.size

            if grad.size == 1:
                grad = xp.broadcast_to(grad, a.shape)
            else:
                axis_ = self.axis if isinstance(self.axis, tuple) else (self.axis,)
                if self.axis is not None:
                    xp = cp.get_array_module(grad)
                    grad = xp.expand_dims(grad, axis_)

                    n = np.prod([a.data.shape[i] for i in axis_])
                
            gr = grad / n

            if a.grad is None:
                a.grad = gr
            else:
                a.grad += gr
