from ..function import Function
from ... import tensor


class Transpose(Function):
    """Function that computes the transpose of a tensor."""

    def __init__(self, a: "tensor.Tensor") -> None:
        self.args = (a,)

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        a = self.args[0]

        if inplace:
            a.data.transpose()
            return a

        self.result = tensor.Tensor(
            a.data.T,
            dtype=a.dtype,
            requires_grad=a.requires_grad,
        )

        return self.result

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            a.grad += grad.T
