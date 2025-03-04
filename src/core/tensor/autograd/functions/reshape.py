import cupy as cp

from ..function import Function
from ... import tensor


class Reshape(Function):
    """Function that reshapes a tensor."""

    __slots__ = ["shape"]

    def __init__(self, a: "tensor.Tensor", *, shape: tuple[int, ...]):
        self.args = (a,)
        self.shape = shape

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        a = self.args[0]
        xp = cp.get_array_module(a.data)

        data = xp.reshape(a.data, self.shape)

        if inplace:
            a.data = data
            return a

        return self._create_output_tensor(data)

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            xp = cp.get_array_module(grad)
            gr = xp.reshape(grad, a.data.shape)

            if a.grad is None:
                a.grad = gr
            else:
                a.grad += gr
