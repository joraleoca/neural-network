import cupy as cp

from ..function import Function
from ... import tensor


class Tanh(Function):
    """Function that computes the element-wise tangent of a tensor."""

    def __init__(self, a: "tensor.Tensor") -> None:
        self.args = (a,)

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        a = self.args[0]
        xp = cp.get_array_module(a.data)

        if inplace:
            a.data[:] = xp.tanh(a.data)
            return a

        return self._create_output_tensor(xp.tanh(a.data))

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            gr = grad * (1 - (self.result.data ** 2))

            if a.grad is None:
                a.grad = gr
            else:
                a.grad += gr
