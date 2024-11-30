from collections.abc import MutableSequence
from collections import deque
from graphlib import TopologicalSorter
from typing import SupportsIndex, TypeVar

import numpy as np
from numpy.typing import NDArray, ArrayLike, DTypeLike

from .autograd import Function
from .autograd import operations as op
from .autograd import functions as func


T = TypeVar("T", bound=np.generic)


class Tensor(MutableSequence[T]):
    """A class representing a multidimensional array (tensor) with support for automatic differentiation."""

    __slots__ = [
        "data",
        "grad",
        "_requires_grad",
        "_grad_operation",
    ]

    data: NDArray[T]
    grad: NDArray[np.floating]

    _requires_grad: bool

    _grad_operation: Function | None

    def __init__(
        self, data: ArrayLike, *, dtype: DTypeLike = None, requires_grad: bool = False
    ):
        """
        Initialize a Tensor object.
        Args:
            data (ArrayLike):
                The input data for the tensor. Can be another Tensor object or any array-like structure.
            dtype (DTypeLike, optional):
                The desired data type for the tensor. If not provided, the data type will be inferred.
            requires_grad (bool, optional):
                If True, the tensor will track gradients for automatic differentiation. Default is False.
        Notes:
            If `data` is another Tensor object, the new tensor will share the same data and gradient properties.
        """
        if isinstance(data, Tensor):
            self.data = data.data.astype(dtype, copy=False)
            self._requires_grad = requires_grad
            return

        self.data = np.array(data, dtype=dtype)
        self._requires_grad = requires_grad

        self._grad_operation = None
        if self._requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.floating)

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

    def __array__(self, dtype: DTypeLike = None) -> NDArray[T]:
        if dtype is None:
            return self.data
        return self.data.astype(dtype)

    def __neg__(self) -> "Tensor[T]":
        return self.apply_operation(op.Neg(self), inplace=False)

    def __add__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self.apply_operation(op.Add(self, other), inplace=False)

    def __radd__(self, other: ArrayLike) -> "Tensor[T]":
        return self.__add__(other)

    def __iadd__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self.apply_operation(op.Add(self, other), inplace=True)

    def __sub__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self.apply_operation(op.Sub(self, other), inplace=False)

    def __rsub__(self, other: ArrayLike):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other.__sub__(self)

    def __isub__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self.apply_operation(op.Sub(self, other), inplace=True)

    def __mul__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self.apply_operation(op.Mul(self, other), inplace=False)

    def __rmul__(self, other: ArrayLike):
        return self.__mul__(other)

    def __imul__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self.apply_operation(op.Mul(self, other), inplace=True)

    def __truediv__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self.apply_operation(op.Div(self, other), inplace=False)

    def __rtruediv__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        return other.__truediv__(self)

    def __itruediv__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self.apply_operation(op.Div(self, other), inplace=True)

    def __pow__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self.apply_operation(op.Pow(self, other), inplace=False)

    def __rpow__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        return other.__pow__(self)

    def __ipow__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self.apply_operation(operation=op.Pow(self, other), inplace=True)

    def __matmul__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self.apply_operation(op.Matmul(self, other), inplace=False)

    def __rmatmul__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        return other.__matmul__(self)

    def __imatmul__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self.apply_operation(op.Matmul(self, other), inplace=True)

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
        return self.apply_operation(inplace=False, operation=op.Abs(self))

    def __str__(self) -> str:
        return str(self.data)

    def __repr__(self) -> str:
        return f"Tensor({self.data}, dtype={self.dtype}, requires_grad={self.requires_grad})"

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, value: T) -> bool:
        return self.data.__contains__(value)

    def __round__(self, decimals: int = 0) -> "Tensor[T]":
        return self.apply_operation(
            inplace=False, operation=func.Round(self, decimals=decimals)
        )

    def __index__(self) -> int:
        if self.size == 1:
            return int(self.data)
        return self.data.__index__()

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool) -> None:
        if self._requires_grad == requires_grad:
            return

        if requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.floating)
        else:
            del self.grad

        self._grad_operation = None
        self._requires_grad = requires_grad

    def item(self) -> T:
        return self.data.item()

    def sum(
        self, axis: tuple[int, ...] | int | None = None, keepdims: bool = False
    ) -> "Tensor[T]" | T:
        """
        Computes the sum of tensor elements over the specified axis.
        Args:
            axis : tuple[int, ...] | int | None, optional
                Axis or axes along which a sum is performed. The default, axis=None, will sum all the elements of the input tensor.
            keepdims : bool, optional
                If True, the reduced axes are left in the result as dimensions with size one. Default is False.
        Returns:
            Tensor[T] | T:
                A tensor with the sum of elements along the specified axis. If no axis is specified, returns the sum of all elements as a scalar.
        """
        return self.apply_operation(
            operation=func.Sum(self, axis=axis, keepdims=keepdims)
        )

    def max(self, axis: SupportsIndex | None = None, keepdims: bool = False):
        """
        Computes the maximum value of tensor elements over the specified axis.
        Args:
            axis : SupportsIndex | None, optional
                Axis or axes along which the maximum is computed. The default, axis=None, will find the maximum element in the tensor.
            keepdims : bool, optional
                If True, the reduced axes are left in the result as dimensions with size one. Default is False.
        Returns:
            Tensor[T] | T:
                A tensor with the maximum value of elements along the specified axis. If no axis is specified, returns the maximum element as a scalar.
        """
        return self.apply_operation(
            operation=func.Max(self, axis=axis, keepdims=keepdims)
        )

    def reshape(self, shape: tuple[int, ...]) -> "Tensor[T]":
        """
        Returns a new tensor with the same data but a different shape.
        Args:
            shape (tuple[int, ...]): The new shape of the tensor.
        Returns:
            Tensor: A new tensor with the specified shape.
        """
        return self.apply_operation(operation=func.Reshape(self, shape=shape))

    def apply_operation(
        self, operation: op.Function, inplace: bool = False
    ) -> "Tensor[T]":
        """
        Applies a specified operation between the current tensor and another tensor.
        Args:
            operation (type[Function]): The operation to apply.
            inplace (bool): If True, the operation is applied in place, modifying the current tensor.
        Returns:
            Tensor: The result of the operation, either a new tensor or the modified current tensor.
        """
        if inplace and self._requires_grad:
            raise ValueError(
                "Inplace operations are not supported for tensors with gradient tracking."
            )

        result = operation(inplace=inplace)

        if result.requires_grad:
            result._grad_operation = operation

        return result

    def backward(self) -> None:
        """
        Computes the gradient of the tensor.
        """
        if not self._requires_grad:
            return

        if np.all(self.grad == 0):
            self.grad = np.ones_like(self.data, dtype=np.floating)

        graph = self._grad_graph()

        graph.prepare()

        while graph:
            for node in graph.get_ready():
                node.backward()
                graph.done(node)

    def _grad_graph(self) -> TopologicalSorter[Function]:
        graph = TopologicalSorter()
        graph.add(self._grad_operation)
        visited: deque[Tensor] = deque([self])

        while visited:
            tensor = visited.popleft()

            if not tensor._grad_operation:
                continue

            for t in tensor._grad_operation.args:
                if t._grad_operation:
                    graph.add(t._grad_operation, tensor._grad_operation)
                    visited.append(t)

        return graph

    def clear_grad(self) -> None:
        """
        Clears the gradient stored in the tensor.
        """
        self.grad = np.zeros_like(self.data, dtype=np.floating)
        self._grad_operation = None

    # The last method so it doesnt interfere with the generic type
    @property
    def T(self) -> "Tensor[T]":
        return self.apply_operation(func.Transpose(self), inplace=False)
