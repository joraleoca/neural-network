from collections.abc import MutableSequence
from typing import SupportsIndex, TypeVar, Any
from collections import deque
from graphlib import TopologicalSorter

import cupy as cp
import numpy as np
from numpy.typing import NDArray, ArrayLike, DTypeLike

from ..config import Config
from .device import Device
from .autograd import (
    Function,
    operations as op,
    functions as func,
)


T = TypeVar("T", bound=cp.generic)


class Tensor(MutableSequence[T]):
    """A class representing a multidimensional array (tensor) with support for automatic differentiation."""

    __slots__ = "data", "grad", "_requires_grad", "_grad_operation", "_device"

    data: NDArray[T]
    grad: NDArray[cp.floating] | None

    _requires_grad: bool

    _grad_operation: Function | None

    _device: Device

    def __init__(
        self,
        data: ArrayLike,
        *,
        dtype: DTypeLike = None,
        requires_grad: bool = False,
        device: str | Device | None = None,
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
            device (str | Device): The device on which to create the tensor.
                Options:
                    - "cpu"    : Create the tensor on the CPU.
                    - "cuda"   : Create the tensor on the GPU (raises an error if not available).
                    - "auto"   : Automatically choose the GPU if available, otherwise use the CPU.
        """
        if device is None:
            device = Config.default_device

        if device == Device.CUDA and not cp.cuda.is_available():
            raise ValueError("Cannot create tensor on 'cuda', GPU is not available.")

        match device:
            case Device.CPU:
                if isinstance(data, np.ndarray):
                    self.data = data if dtype is None else data.astype(dtype)
                else:
                    self.data = np.array(data, dtype=dtype or Config.default_dtype)
            case Device.CUDA:
                if isinstance(data, cp.ndarray):
                    self.data = data if dtype is None else data.astype(dtype)   #type: ignore
                else:
                    self.data = cp.array(data, dtype=dtype or Config.default_dtype)
            case _:
                raise ValueError(f"Unknown device {device}")

        self._requires_grad = requires_grad
        self._grad_operation = None
        self.grad = None
        self._device = Device(device)

    """Magic methods"""

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> "Tensor[T] | T":
        return self.apply_operation(op.Index(self, idx))

    def __setitem__(self, idx, value) -> None:
        if self.requires_grad:
            raise ValueError(
                "Tensor can only be modified in-place when it doesn't require grad"
            )

        self.data[idx] = value

    def __delitem__(self, idx) -> None:
        raise NotImplementedError("Deletion not supported for Tensor")

    def insert(self, idx, value) -> None:
        raise NotImplementedError("Insertion not supported for Tensor")

    def __array__(self, dtype: DTypeLike = None) -> NDArray[T]:
        if isinstance(self.data, np.ndarray):
            data = self.data
        else:
            data = cp.ndarray.get(self.data)
 
        return data.astype(dtype)

    def __neg__(self) -> "Tensor[T]":
        return self.apply_operation(op.Neg(self), inplace=False)

    def __add__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        return self.apply_operation(op.Add(self, other), inplace=False)

    def __radd__(self, other: ArrayLike) -> "Tensor[T]":
        return self.__add__(other)

    def __iadd__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        return self.apply_operation(op.Add(self, other), inplace=True)

    def __sub__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        return self.apply_operation(op.Sub(self, other), inplace=False)

    def __rsub__(self, other: ArrayLike):
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        return other.__sub__(self)

    def __isub__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        return self.apply_operation(op.Sub(self, other), inplace=True)

    def __mul__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        return self.apply_operation(op.Mul(self, other), inplace=False)

    def __rmul__(self, other: ArrayLike):
        return self.__mul__(other)

    def __imul__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        return self.apply_operation(op.Mul(self, other), inplace=True)

    def __truediv__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        return self.apply_operation(op.Div(self, other), inplace=False)

    def __rtruediv__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        return other.__truediv__(self)

    def __itruediv__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        return self.apply_operation(op.Div(self, other), inplace=True)

    def __pow__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        return self.apply_operation(op.Pow(self, other), inplace=False)

    def __rpow__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        return other.__pow__(self)

    def __ipow__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        return self.apply_operation(operation=op.Pow(self, other), inplace=True)

    def __matmul__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        return self.apply_operation(op.Matmul(self, other), inplace=False)

    def __rmatmul__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        return other.__matmul__(self)

    def __imatmul__(self, other: ArrayLike) -> "Tensor[T]":
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        return self.apply_operation(op.Matmul(self, other), inplace=True)

    def __eq__(self, other: ArrayLike) -> bool:
        xp = cp.get_array_module(self.data)
        other = xp.array(other)
        return xp.array_equal(self.data, other)

    def __ne__(self, other: ArrayLike) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other: ArrayLike) -> NDArray[cp.bool_]:
        xp = cp.get_array_module(self.data)
        other = xp.array(other)
        return xp.less(self.data, other)

    def __le__(self, other: ArrayLike) -> NDArray[cp.bool_]:
        xp = cp.get_array_module(self.data)
        other = xp.array(other)
        return xp.less_equal(self.data, other)

    def __gt__(self, other: ArrayLike) -> NDArray[cp.bool_]:
        xp = cp.get_array_module(self.data)
        other = xp.array(other)
        return xp.greater(self.data, other)

    def __ge__(self, other: ArrayLike) -> NDArray[cp.bool_]:
        xp = cp.get_array_module(self.data)
        other = xp.array(other)
        return xp.greater_equal(self.data, other)

    def __abs__(self) -> "Tensor[T]":
        return self.apply_operation(inplace=False, operation=op.Abs(self))

    def __str__(self) -> str:
        return str(self.data)

    def __repr__(self) -> str:
        return f"Tensor({self.data}, dtype={self.dtype}, requires_grad={self.requires_grad})"

    def __iter__(self):
        # TODO: Make the iter return Tensors
        return iter(self.data)

    def __contains__(self, value: T) -> bool:
        return self.data.__contains__(value)

    def __round__(self, decimals: int = 0) -> "Tensor[T]":
        return self.apply_operation(
            inplace=False, operation=func.Round(self, decimals=decimals)
        )

    def __index__(self) -> int:
        return self.data.__index__()    #type: ignore

    def __copy__(self):
        """Return a shallow copy of this tensor."""
        return Tensor(
            self.data.copy(), dtype=self.dtype, requires_grad=self.requires_grad
        )

    def copy(self) -> "Tensor[T]":
        """Return a shallow copy of this tensor."""
        return self.__copy__()

    """Properties"""

    @property
    def dtype(self) -> cp.dtype:
        """Returns the data type (dtype) of the tensor."""
        return self.data.dtype

    @dtype.setter
    def dtype(self, dtype: DTypeLike) -> None:
        """Sets the data type (dtype) of the tensor."""
        self.data = self.data.astype(dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the tensor."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the tensor."""
        return self.data.ndim

    @property
    def size(self) -> int:
        """Returns the number of elements in the tensor."""
        return self.data.size
    
    @property
    def strides(self) -> tuple[int, ...]:
        """Returns the strides of the tensor."""
        return self.data.strides

    @property
    def requires_grad(self) -> bool:
        """
        Gets or sets the flag indicating whether the tensor requires gradient tracking.
        """
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool) -> None:
        if self._requires_grad == requires_grad:
            return

        self.grad = None
        self._grad_operation = None
        self._requires_grad = requires_grad

    @property
    def device(self) -> str:
        """
        Returns the device on which the data is stored.
        """
        return self._device.value

    """Utils"""

    def item(self) -> T:
        """Returns the value of the tensor as a standard Python scalar."""
        return self.data.item()

    def sum(
        self, axis: tuple[int, ...] | int | None = None, keepdims: bool = False
    ) -> "Tensor[T] | T":
        """
        Computes the sum of tensor elements over the specified axis.

        Args:
            axis: tuple[int, ...] | int | None, optional
                Axis or axes along which a sum is performed. The default, axis=None, will sum all the elements of the input tensor.
            keepdims: bool, optional
                If True, the reduced axes are left in the result as dimensions with size one. Default is False.
        Returns:
            Tensor[T] | T:
                A tensor with the sum of elements along the specified axis. If no axis is specified, returns the sum of all elements as a scalar.
        """
        return self.apply_operation(
            operation=func.Sum(self, axis=axis, keepdims=keepdims)
        )

    def max(
        self, axis: SupportsIndex | None = None, keepdims: bool = False
    ) -> "Tensor[T] | T":
        """
        Computes the maximum value of tensor elements over the specified axis.

        Args:
            axis : SupportsIndex | None, optional
                Axis or axes along which the maximum is computed.
                The default, axis=None, will find the maximum element in the tensor.
            keepdims : bool, optional
                If True, the reduced axes are left in the result as dimensions with size one. Default is False.
        Returns:
            Tensor[T] | T:
                A tensor with the maximum value of elements along the specified axis.
                If no axis is specified, returns the maximum element as a scalar.
        """
        return self.apply_operation(
            operation=func.Max(self, axis=axis, keepdims=keepdims)
        )

    def min(
        self, axis: SupportsIndex | None = None, keepdims: bool = False
    ) -> "Tensor[T] | T":
        """
        Computes the minimum value of tensor elements over the specified axis.

        Args:
            axis : SupportsIndex | None, optional
                Axis or axes along which the minimum is computed.
                The default, axis=None, will find the minimum element in the tensor.
            keepdims : bool, optional
                If True, the reduced axes are left in the result as dimensions with size one. Default is False.
        Returns:
            Tensor[T] | T:
                A tensor with the minimum value of elements along the specified axis.
                If no axis is specified, returns the minimum element as a scalar.
        """
        return self.apply_operation(
            operation=func.Min(self, axis=axis, keepdims=keepdims)
        )

    def mean(self, axis: int | tuple[int, ...] | None = None) -> "Tensor[T] | T":
        """
        Computes the mean value of tensor elements over the specified axis.

        Args:
            axis: int | tuple[int, ...] | None, optional
                Axis or axes along which the mean is computed.
                The default, axis=None, will find the mean element in the tensor.
        Returns:
            Tensor[T] | T:
                A tensor with the mean value of elements along the specified axis.
                If no axis is specified, returns the mean element as a scalar.
        """
        return self.apply_operation(operation=func.Mean(self, axis=axis))
    
    def argmax(self, *, axis: SupportsIndex | None = None, dtype: DTypeLike, out: None = None, keepdims: bool = False) -> "Tensor[T]":
        """
        Returns the indices of the maximum values along an axis.

        Args:
            axis (SupportsIndex | None): The axis or axes along which to perform the operation.
            dtype (DTypeLike): The data type of the returned tensor.
            out (None): Not supported. Made for compatibility.
            keepdims (bool): If True, the reduced axes are left in the result as dimensions with size one. Default is False.
        Returns:
            Tensor: A tensor with the indices of the maximum values along the specified axis.
        """
        out_ = self.apply_operation(operation=func.Argmax(self, axis=axis, keepdims=keepdims))
        out_.dtype = dtype
        return out_

    def reshape(self, shape: tuple[int, ...], *, inplace: bool = True) -> "Tensor[T]":
        """
        Returns a new tensor with the same data but a different shape.

        Args:
            shape (tuple[int, ...]): The new shape of the tensor.
        Returns:
            Tensor: A new tensor with the specified shape.
        """
        return self.apply_operation(operation=func.Reshape(self, shape=shape), inplace=inplace)

    def flatten(self) -> "Tensor[T]":
        """
        Return a new Tensor with the same data but as an 1D array.

        Returns:
            Tensor: A new 1D tensor.
        """
        return self.apply_operation(operation=func.Flatten(self))

    def fill(self, value: Any) -> None:
        if self.requires_grad:
            raise ValueError("Fill cannot be done when requires_grad is True")
        self.data.fill(value)

    """Autograd"""

    def apply_operation(
        self, operation: Function, inplace: bool = False
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
        Computes the gradient of the tensor.\n
        The tensor must have requires_grad set to True.\n
        The gradient is stored in the grad attribute.\n
        """
        if not self._requires_grad:
            return

        if self.grad is None:
            xp = cp.get_array_module(self.data)
            
            if self.data.dtype.kind == "f":
                dtype_ = xp.dtype(self.data.dtype)
            elif Config.default_dtype.kind == "f":
                dtype_ = Config.default_dtype
            else:
                dtype_ = np.float32

            self.grad = xp.ones_like(self.data, dtype=dtype_)   #type: ignore

        graph = self._grad_graph()

        graph.prepare()

        while graph:
            for node in graph.get_ready():
                node.backward()
                graph.done(node)

    def _grad_graph(self) -> TopologicalSorter[Function]:
        """
        Creates a graph of the gradient operations for the tensor.

        Returns:
            TopologicalSorter[Function]: A graph of the gradient operations.
        """
        graph = TopologicalSorter()
        graph.add(self._grad_operation)
        to_visit: deque[Tensor] = deque([self])

        while to_visit:
            tensor: Tensor = to_visit.popleft()

            if not tensor._grad_operation:
                continue

            for t in tensor._grad_operation.args:
                if t._grad_operation:
                    graph.add(t._grad_operation, tensor._grad_operation)
                    to_visit.append(t)

            tensor._grad_operation = None

        return graph

    def clear_grad(self) -> None:
        """
        Clears the gradient stored in the tensor.
        """
        if self._requires_grad and self.grad is not None and self.grad.size != 1:
            self.grad.fill(0)
        else:
            self.grad = None

        self._grad_operation = None

    # The last method so it does not interfere with the generic type
    @property
    def T(self) -> "Tensor[T]":
        """Returns the transpose of the tensor."""
        return self.apply_operation(func.Transpose(self), inplace=False)
