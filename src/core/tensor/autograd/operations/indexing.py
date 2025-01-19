from copy import deepcopy

import numpy as np

from ..function import Function
from ... import tensor


class Index(Function):
    """Function that index a tensor."""

    __slots__ = "idx"

    def __init__(self, a: "tensor.Tensor", idx) -> None:
        self.args = (a,)
        self.idx = idx

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        if inplace:
            raise ValueError("Index cannot be inplace.")

        a = self.args[0]

        self.result = tensor.Tensor(a.data[self.idx], dtype=a.dtype, requires_grad=a.requires_grad)

        return self.result

    def backward(self) -> None:
        a = self.args[0]
        assert self.result is not None, "Result cannot be None."
        grad = self.result.grad

        if a.requires_grad:
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
                a.grad[self.idx] = deepcopy(grad)
            else:
                a.grad[self.idx] += grad
