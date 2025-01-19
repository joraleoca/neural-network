import numpy as np

from ..function import Function
from ... import tensor


class Tanh(Function):
    """Function that computes the element-wise tangent of a tensor."""

    def __init__(self, a: "tensor.Tensor") -> None:
        self.args = (a,)

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        a = self.args[0]

        if inplace:
            a.data[:] = np.tanh(a.data)
            return a

        self.result = tensor.Tensor(
            np.tanh(a.data),
            dtype=a.dtype,
            requires_grad=a.requires_grad,
        )

        return self.result

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            gr = grad * (1 - (np.tanh(a) ** 2))

            if a.grad is None:
                a.grad = gr
            else:
                a.grad += gr
