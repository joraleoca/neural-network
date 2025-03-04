from ..function import Function
from ... import tensor
from src.core.tensor.autograd.utils import update_tensor_grad


class Sub(Function):
    """Function that computes the element-wise subtraction of two tensors."""

    def __init__(self, a: "tensor.Tensor", b: "tensor.Tensor") -> None:
        self.args = (a, b)

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        a, b = self.args

        if inplace:
            a.data[:] -= b.data
            return a

        return self._create_output_tensor(a.data - b.data)

    def backward(self) -> None:
        a, b = self.args
        grad = self.result.grad

        if a.requires_grad:
            update_tensor_grad(a, grad)

        if b.requires_grad:
            update_tensor_grad(b, -grad)
