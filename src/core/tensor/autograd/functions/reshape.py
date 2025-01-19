import numpy as np

from ..function import Function
from ... import tensor


class Reshape(Function):
    """Function that reshapes a tensor."""

    __slots__ = ["shape"]

    def __init__(self, a: "tensor.Tensor", *, shape: tuple[int, ...]):
        self.args = (a,)
        self.shape = shape

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        a = self.args[0]

        if inplace:
            a.data.reshape(self.shape)
            return a

        self.result = tensor.Tensor(
            np.reshape(a.data, self.shape),
            dtype=a.dtype,
            requires_grad=a.requires_grad,
        )

        return self.result

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            gr = np.reshape(grad, a.data.shape)

            if a.grad is None:
                a.grad = gr
            else:
                a.grad += gr
