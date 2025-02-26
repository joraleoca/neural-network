from copy import deepcopy

import cupy as cp

from ..function import Function
from ... import tensor


class As_Strided(Function):
    """Creates a view of a tensor with the specified strides and shape."""
    
    __slots__ = "shape", "strides"

    def __init__(self, a: "tensor.Tensor", *, shape: tuple[int], strides: tuple[int]) -> None:
        self.args = (a,)
        self.shape = shape
        self.strides = strides

    def __call__(self, *, inplace: bool = True) -> "tensor.Tensor":
        if inplace:
            raise NotImplementedError("Not inplace as_strided is not supported. It always returns a view.")

        a = self.args[0]

        xp = cp.get_array_module(a.data)

        self.result = tensor.Tensor(
            xp.lib.stride_tricks.as_strided(a.data, self.shape, self.strides),
            requires_grad=a.requires_grad,
        )

        return self.result

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            xp = cp.get_array_module(a.data)
            gr = xp.lib.stride_tricks.as_strided(grad, a.shape, a.strides)

            if a.grad is None:
                a.grad = deepcopy(gr)
            else:
                a.grad += gr
