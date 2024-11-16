from collections.abc import MutableSequence
from collections import deque
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray, ArrayLike, DTypeLike

import autograd
import autograd.operations as op


T = TypeVar("T", bound=np.generic)


class Tensor(MutableSequence[T]):
    """A class representing a multi-dimensional array (tensor) with support for automatic differentiation."""

    __slots__ = [
        "data",
        "grad",
        "requires_grad",
        "_grad_stack",
    ]

    data: NDArray[T]

    grad: NDArray[T]

    requires_grad: bool

    _grad_stack: deque[autograd.Context]

    def __init__(
        self, data: ArrayLike, *, dtype: DTypeLike = None, requires_grad: bool = False
    ):
        self.data = np.array(data, dtype=dtype)
        self.requires_grad = requires_grad

        if self.requires_grad:
            self._grad_stack = deque()
            self.grad = np.zeros_like(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> "Tensor[T]" | T:
        item = self.data[idx]

        if isinstance(item, np.ndarray):
            return Tensor(item, dtype=self.dtype, requires_grad=self.requires_grad)

        return item

    def __setitem__(self, idx, value) -> None:
        self.data[idx] = value

    def __delitem__(self, idx) -> None:
        raise NotImplementedError("Deletion not supported for Tensor")

    def insert(self, idx, value) -> None:
        raise NotImplementedError("Insertion not supported for Tensor")

    def __array__(self) -> NDArray[T]:
        return self.data

    def __neg__(self) -> "Tensor[T]":
        return self.apply_operation(inplace=False, operation=op.Neg)

    def __add__(self, other: ArrayLike) -> "Tensor[T]":
        return self.apply_operation(other, inplace=False, operation=op.Add)

    def __radd__(self, other: ArrayLike) -> "Tensor[T]":
        return self.__add__(other)

    def __iadd__(self, other: ArrayLike) -> "Tensor[T]":
        return self.apply_operation(other, inplace=True, operation=op.Add)

    def __sub__(self, other: ArrayLike) -> "Tensor[T]":
        return self.apply_operation(other, inplace=False, operation=op.Sub)

    def __rsub__(self, other: ArrayLike):
        other = self._arr_to_tensor(other)
        return other.__sub__(self)

    def __isub__(self, other: ArrayLike) -> "Tensor[T]":
        return self.apply_operation(other, inplace=True, operation=op.Sub)

    def __mul__(self, other: ArrayLike) -> "Tensor[T]":
        return self.apply_operation(other, inplace=False, operation=op.Mul)

    def __rmul__(self, other: ArrayLike):
        return self.__mul__(other)

    def __imul__(self, other: ArrayLike) -> "Tensor[T]":
        return self.apply_operation(other, inplace=True, operation=op.Mul)

    def __truediv__(self, other: ArrayLike) -> "Tensor[T]":
        return self.apply_operation(other, inplace=False, operation=op.Div)

    def __rtruediv__(self, other: ArrayLike) -> "Tensor[T]":
        other = self._arr_to_tensor(other)
        return other.__truediv__(self)

    def __itruediv__(self, other: ArrayLike) -> "Tensor[T]":
        return self.apply_operation(other, inplace=True, operation=op.Div)

    def __pow__(self, other: ArrayLike) -> "Tensor[T]":
        return self.apply_operation(other, inplace=False, operation=op.Pow)

    def __rpow__(self, other: ArrayLike) -> "Tensor[T]":
        other = self._arr_to_tensor(other)
        return other.__pow__(self)

    def __ipow__(self, other: ArrayLike) -> "Tensor[T]":
        return self.apply_operation(other, inplace=True, operation=op.Pow)

    def __eq__(self, other: ArrayLike) -> bool:
        return np.array_equal(self.data, other)

    def __ne__(self, other: ArrayLike) -> bool:
        return not np.array_equal(self.data, other)

    def __lt__(self, other: ArrayLike) -> NDArray[np.bool_]:
        return np.less(self.data, other)

    def __le__(self, other: ArrayLike) -> NDArray[np.bool_]:
        return np.less_equal(self.data, other)

    def __gt__(self, other: ArrayLike) -> NDArray[np.bool_]:
        return np.greater(self.data, other)

    def __ge__(self, other: ArrayLike) -> NDArray[np.bool_]:
        return np.greater_equal(self.data, other)

    def __abs__(self) -> "Tensor[T]":
        return self.apply_operation(inplace=False, operation=op.Abs)

    def __str__(self) -> str:
        return str(self.data)

    def __repr__(self) -> str:
        return f"Tensor({self.data}, dtype={self.dtype}, requires_grad={self.requires_grad})"

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, value: T) -> bool:
        return self.data.__contains__(value)

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    def apply_operation(
        self, *args: ArrayLike, inplace: bool, operation: type[op.Function]
    ) -> "Tensor[T]":
        """
        Applies a specified operation between the current tensor and another tensor.
        Args:
            args (ArrayLike): The others tensors to apply the operation with.
            inplace (bool): If True, the operation is applied in place, modifying the current tensor.
            operation (type[Function]): The operation to apply.
        Returns:
            Tensor: The result of the operation, either a new tensor or the modified current tensor.
        """
        operands = [self] + [self._arr_to_tensor(o) for o in args]

        ctx = autograd.Context()

        result = operation.forward(ctx, *operands, inplace=inplace)

        if result.requires_grad:
            result._grad_stack.append(ctx)

        return result

    def _arr_to_tensor(self, arr: ArrayLike) -> "Tensor[T]":
        """
        Converts an array-like object to a Tensor. No grad.
        Args:
            arr (ArrayLike): The array-like object to be converted.
        Returns:
            Tensor: The converted Tensor object. If the input is already a Tensor, it is returned as is.
        """
        if not isinstance(arr, Tensor):
            return Tensor(arr, dtype=self.dtype, requires_grad=False)
        return arr

    def gradient(self) -> None:
        """
        Computes the gradient of the tensor.
        This method calculates the gradient of the tensor by applying the chain rule
        through the computational graph. If the gradient is initially zero, it sets
        it to ones. It then iterates through the gradient stack, applying the
        backward function for each context and recursively computing gradients for
        dependent tensors.
        """
        if np.all(self.grad == 0):
            self.grad = np.ones_like(self.grad)

        while self._grad_stack:
            ctx = self._grad_stack.pop()

            ctx.backwards_func(ctx, self.grad)

            for d in ctx.data:
                if d is not self and d.requires_grad:
                    d.gradient()

    def clear_grad(self) -> None:
        """
        Clears the gradient stored in the tensor.
        """
        self.grad = np.zeros_like(self.data)
