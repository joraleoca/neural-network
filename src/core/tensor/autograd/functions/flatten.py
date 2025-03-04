import cupy as cp

from ..function import Function
from ... import tensor


class Flatten(Function):
    """Flatten a tensor."""

    def __init__(self, a: "tensor.Tensor"):
        self.args = (a,)

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        if inplace:
            raise NotImplementedError("Inplace flatten is not supported.")

        a = self.args[0]

        return self._create_output_tensor(a.data.flatten())

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            xp = cp.get_array_module(grad)
            gr = xp.reshape(grad, a.shape)

            if a.grad is None:
                a.grad = gr
            else:
                a.grad += grad.reshape(a.shape)
