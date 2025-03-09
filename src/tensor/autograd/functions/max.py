from typing import SupportsIndex, Sequence

import cupy as cp

from ..function import Function
from ... import tensor


class Max(Function):
    """Function that computes the maximum value of a tensor."""

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

        return self._create_output_tensor(
            a.data.max(axis=self.axis, keepdims=self.keepdims)
        )

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            mask = a.data == a.data.max(axis=self.axis, keepdims=True)

            if self.axis is not None and not self.keepdims:
                xp = cp.get_array_module(grad)
                grad = xp.expand_dims(grad, self.axis)

            gr = grad * mask

            if a.grad is None:
                a.grad = gr
            else:
                a.grad += gr
