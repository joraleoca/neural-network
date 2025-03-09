import logging

import cupy as cp

from src.constants import EPSILON

from ... import tensor
from ..function import Function
from ..utils import update_tensor_grad


class Pow(Function):
    """Function that computes the element-wise power of two tensors."""

    def __init__(self, a: "tensor.Tensor", b: "tensor.Tensor") -> None:
        self.args = (a, b)

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        a, b = self.args

        if inplace:
            a.data[:] **= b.data
            return a

        return self._create_output_tensor(a.data**b.data)

    def backward(self) -> None:
        a, b = self.args
        grad = self.result.grad

        if a.requires_grad:
            gr = grad * (b.data * a.data ** (b.data - 1))

            update_tensor_grad(a, gr)

        if b.requires_grad:
            xp = cp.get_array_module(a.data)
            if not xp.all(a.data > 0):
                logging.warning(
                    f"Cannot compute gradient with respect to b for a={a.data} <= 0. Gradient not modified."
                )
                return

            xp = cp.get_array_module(a.data, b.data)
            gr = grad * (a.data**b.data * xp.log(a.data + EPSILON))

            update_tensor_grad(b, gr)
