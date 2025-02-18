from copy import deepcopy

from ..function import Function
from ... import tensor


class Compose(Function):
    """Compose function."""

    def __init__(self, tensors: "list[tensor.Tensor] | tuple[tensor.Tensor, ...]") -> None:
        self.args = tuple(tensors)

        shape = tensors[0].shape

        if not all(shape == t.shape for t in tensors):
            raise ValueError("All tensors must be of the same shape")

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        if inplace:
            raise NotImplementedError("Inplace compose is not supported.")

        self.result = tensor.Tensor(
            self.args,
            requires_grad=any(t.requires_grad for t in self.args),
        )

        return self.result

    def backward(self) -> None:
        grad = self.result.grad

        for i, t in enumerate(self.args):
            if t.requires_grad:
                if t.grad is None:
                    t.grad = deepcopy(grad[i])
                else:
                    t.grad += grad[i]
