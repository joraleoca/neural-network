from src.constants import EPSILON

from ... import tensor
from ..function import Function
from ..utils import update_tensor_grad


class Div(Function):
    """Function that computes the element-wise division of two tensors."""

    def __init__(self, a: "tensor.Tensor", b: "tensor.Tensor") -> None:
        self.args = (a, b)

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        a, b = self.args

        if inplace:
            a.data[:] /= b.data + EPSILON
            return a

        return self._create_output_tensor(a.data / (b.data + EPSILON))

    def backward(self) -> None:
        a, b = self.args
        grad = self.result.grad

        if a.requires_grad:
            gr = grad / (b.data + EPSILON)

            update_tensor_grad(a, gr)

        if b.requires_grad:
            gr = grad * -a.data / ((b.data**2) + EPSILON)

            update_tensor_grad(b, gr)
