import numpy as np

from ..function import Function
from ... import tensor


class Log(Function):
    def __init__(self, a: "tensor.Tensor"):
        self.args = (a,)

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        a = self.args[0]

        if inplace:
            a.data[:] = np.log(a.data)
            return a

        self.result = tensor.Tensor(
            np.log(a.data),
            dtype=a.dtype,
            requires_grad=a.requires_grad,
        )

        return self.result

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            a.grad += grad * 1 / a.data
