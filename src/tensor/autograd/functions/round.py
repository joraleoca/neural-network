from copy import deepcopy

import cupy as cp

from ..function import Function
from ... import tensor


class Round(Function):
    """Function that rounds the elements of a tensor to the nearest integer."""

    __slots__ = ["decimals"]

    def __init__(self, a: "tensor.Tensor", *, decimals: int = 0):
        self.args = (a,)
        self.decimals = decimals

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        a = self.args[0]
        xp = cp.get_array_module(a.data)

        data = xp.round(a.data, self.decimals)

        if inplace:
            a.data = data
            return a

        return self._create_output_tensor(data)

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            # Not differentiable, but we need to propagate the gradient
            if a.grad is None:
                a.grad = deepcopy(grad)
            else:
                a.grad += grad
