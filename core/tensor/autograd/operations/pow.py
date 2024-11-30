import numpy as np

from ..function import Function
from ... import tensor


class Pow(Function):
    """Function that computes the element-wise power of two tensors."""

    def __init__(self, a: "tensor.Tensor", b: "tensor.Tensor") -> None:
        self.args = (a, b)

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        a, b = self.args

        if inplace:
            a.data[:] **= b.data
            return a

        self.result = tensor.Tensor(
            a.data**b.data,
            dtype=a.dtype if a.dtype == b.dtype else np.floating,
            requires_grad=a.requires_grad or b.requires_grad,
        )

        return self.result

    def backward(self) -> None:
        a, b = self.args
        grad = self.result.grad

        if a.requires_grad:
            a_grad = grad * (b.data * a.data ** (b.data - 1))

            if a.size == 1:
                a.grad = np.full_like(a_grad, a.grad.item())

            a.grad += a_grad

        if b.requires_grad:
            if not np.all(a.data > 0):
                print(
                    f"Warning: Cannot compute gradient with respect to b for a={a.data} <= 0. Gradient not modified."
                )
                return

            b_grad = grad * (a.data**b.data * np.log(a.data))

            if b.size == 1:
                b.grad = np.full_like(b_grad, b.grad.item())

            b.grad += b_grad
