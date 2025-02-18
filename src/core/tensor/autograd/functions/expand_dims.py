import cupy as cp

from ..function import Function
from ... import tensor


class ExpandDims(Function):
    """Expand dims function."""

    __slots__ = ["axis"]

    def __init__(
        self, a: "tensor.Tensor", axis: int | list[int] | tuple[int, ...]
    ) -> None:
        self.args = (a,)
        self.axis = axis

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        if inplace:
            raise NotImplementedError("Inplace expand dims is not supported.")

        a = self.args[0]
        xp = cp.get_array_module(a.data)

        self.result = tensor.Tensor(
            xp.expand_dims(a.data, self.axis),
            requires_grad=a.requires_grad,
        )

        return self.result

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            xp = cp.get_array_module(grad)
            gr = xp.reshape(grad, a.shape)

            if a.grad is None:
                a.grad = gr
            else:
                a.grad += gr
