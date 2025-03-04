from ..function import Function
from ... import tensor


class Abs(Function):
    """Function that computes the element-wise absolute value of a tensor."""

    def __init__(self, a: "tensor.Tensor") -> None:
        self.args = (a,)

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        a = self.args[0]

        if inplace:
            a.data[:] = abs(a.data)
            return a

        return self._create_output_tensor(abs(a.data))

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            # -1 for negative, 1 for positive, 0 for 0
            gr = grad * ((a.data > 0) - grad * (a.data < 0)) * (a.data != 0)

            if a.grad is None:
                a.grad = gr
            else:
                a.grad += gr