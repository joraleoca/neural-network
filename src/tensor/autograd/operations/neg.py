from ..function import Function
from ... import tensor


class Neg(Function):
    """Function that computes the element-wise negation of a tensor."""

    def __init__(self, a: "tensor.Tensor") -> None:
        self.args = (a,)

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        a = self.args[0]

        if inplace:
            a.data[:] = -a.data
            return a


        return self._create_output_tensor(-a.data)


    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            if a.grad is None:
                a.grad = -grad
            else:
                a.grad -= grad
