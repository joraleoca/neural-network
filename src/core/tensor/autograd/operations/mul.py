from ..function import Function
from ... import tensor
from src.core.tensor.autograd.utils import update_tensor_grad


class Mul(Function):
    """Function that computes the element-wise multiplication of two tensors."""

    def __init__(self, a: "tensor.Tensor", b: "tensor.Tensor") -> None:
        self.args = (a, b)

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        a, b = self.args

        if inplace:
            a.data[:] *= b.data
            return a

        return self._create_output_tensor(a.data * b.data)

    def backward(self) -> None:
        a, b = self.args
        grad = self.result.grad

        if a.requires_grad:
            gr = grad * b.data

            update_tensor_grad(a, gr)

        if b.requires_grad:
            gr = grad * a.data

            update_tensor_grad(b, gr)
