import numpy as np

from ..function import Function
from ... import tensor


class Sum(Function):
    __slots__ = ["axis", "keepdims"]

    def __init__(
        self, a, *, axis: tuple[int, ...] | int | None = None, keepdims: bool = False
    ):
        self.args = (a,)
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        if inplace:
            raise NotImplementedError("Inplace addition is not supported.")

        a = self.args[0]

        self.result = tensor.Tensor(
            np.sum(a.data, axis=self.axis, keepdims=self.keepdims),
            dtype=a.dtype,
            requires_grad=a.requires_grad,
        )

        return self.result

    def backward(self) -> None:
        a = self.args[0]
        if not a.requires_grad:
            return

        grad = self.result.grad

        # Handle scalar gradient (from global sum)
        if np.isscalar(grad) or grad.ndim == 0 or self.keepdims:
            a.grad += grad * np.ones_like(a.data)  # type: ignore
            return

        # Handle axis-specific sums
        grad_shape = list(a.data.shape)
        if self.axis is not None:
            if isinstance(self.axis, int):
                grad_shape[self.axis] = 1
            else:
                for ax in self.axis:
                    grad_shape[ax] = 1
        a.grad += grad.reshape(grad_shape) * np.ones_like(a.data)
