import numpy as np
import logging

from ..function import Function
from ... import tensor
from ..utils import update_tensor_grad
from src.constants import EPSILON


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
            gr = grad * (b.data * a.data ** (b.data - 1))

            update_tensor_grad(a, gr)

        if b.requires_grad:
            if not np.all(a.data > 0):
                logging.warning(
                    f"Cannot compute gradient with respect to b for a={a.data} <= 0. Gradient not modified."
                )
                return

            gr = grad * (a.data**b.data * np.log(a.data + EPSILON))

            update_tensor_grad(b, gr)
