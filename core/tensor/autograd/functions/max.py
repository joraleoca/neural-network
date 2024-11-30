from typing import SupportsIndex, Sequence

import numpy as np

from ..function import Function
from ... import tensor


class Max(Function):
    __slots__ = ["axis", "keepdims"]

    def __init__(
        self,
        a: "tensor.Tensor",
        *,
        axis: SupportsIndex | Sequence[SupportsIndex] | None = None,
        keepdims: bool = False,
    ):
        self.args = (a,)
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        if inplace:
            raise NotImplementedError("Inplace max is not supported.")

        a = self.args[0]

        self.result = tensor.Tensor(
            a.data.max(axis=self.axis, keepdims=self.keepdims),
            dtype=a.dtype,
            requires_grad=a.requires_grad,
        )

        return self.result

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            mask = a.data == a.data.max(axis=self.axis, keepdims=True)

            if self.axis and not self.keepdims:
                grad = np.expand_dims(grad, self.axis)

            a.grad += grad * mask
