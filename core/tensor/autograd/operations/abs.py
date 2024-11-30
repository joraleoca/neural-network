from ..function import Function
from ... import tensor


class Abs(Function):
    def __init__(self, a: "tensor.Tensor") -> None:
        self.args = (a,)

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        a = self.args[0]

        if inplace:
            a.data[:] = abs(a.data)
            return a

        self.result = tensor.Tensor(
            abs(a.data),
            dtype=a.dtype,
            requires_grad=a.requires_grad,
        )

        return self.result

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            # -1 for negative, 1 for positive, 0 for 0
            a.grad += grad * ((a.data > 0) - grad * (a.data < 0)) * (a.data != 0)
