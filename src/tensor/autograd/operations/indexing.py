from copy import deepcopy

import cupy as cp

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

        return self._create_output_tensor(a.data[self.idx], a.dtype)

    def backward(self) -> None:
        a = self.args[0]
        assert self.result is not None, "Result cannot be None."
        grad = self.result.grad

        if grad.size == 1:
            grad = grad.item()

        if a.requires_grad:
            if a.grad is None:
                xp = cp.get_array_module(a.data)
                a.grad = xp.zeros_like(a.data)
                a.grad[self.idx] = deepcopy(grad)
            else:
                a.grad[self.idx] += grad
