import numpy as np

from ..function import Function
from ... import tensor


class Pad(Function):
    """Pad a tensor."""

    __slots__ = ["original_shape", "pad_width", "value"]

    def __init__[T](
        self, a: "tensor.Tensor[T]", pad_width: int | tuple = 0, *, value: T = 0
    ) -> None:
        self.args = (a,)
        self.original_shape = a.shape
        self.pad_width = pad_width
        self.value = value

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        if inplace:
            raise NotImplementedError("Inplace padding is not supported.")

        a = self.args[0]

        self.result = tensor.Tensor(
            np.pad(
                a, pad_width=self.pad_width, mode="constant", constant_values=self.value
            ),
            dtype=a.dtype,
            requires_grad=a.requires_grad,
        )

        return self.result

    def backward(self) -> None:
        a = self.args[0]
        grad = self.result.grad

        if a.requires_grad:
            gr = grad[_index(self.pad_width)]

            if a.grad is None:
                a.grad = gr
            else:
                a.grad += gr

def _index(pad_width: int | tuple) -> tuple[slice, ...]:
    if isinstance(pad_width, int):
        return (slice(pad_width, -pad_width if pad_width > 0 else None),)

    if isinstance(pad_width[0], int):
        return (slice(pad_width[0], -pad_width[1] if pad_width[1] > 0 else None),)

    if isinstance(pad_width[0], tuple):
        return tuple(slice(pad[0], -pad[1] if pad[1] > 0 else None) for pad in pad_width)

    raise ValueError("Invalid pad_width format")