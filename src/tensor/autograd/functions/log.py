import cupy as cp

from ..function import Function
from ... import tensor


class Log(Function):
    """Function that computes the element-wise natural logarithm of a tensor."""

    def __init__(self, a: "tensor.Tensor"):
        self.args = (a,)

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        a = self.args[0]
        xp = cp.get_array_module(a.data)

        data = xp.log(a.data)

        if inplace:
            a.data = data
            return a

        return self._create_output_tensor(data)

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            gr = grad * 1 / a.data

            if a.grad is None:
                a.grad = gr
            else:
                a.grad += grad * 1 / a.data
