from typing import Iterable, Sequence, SupportsIndex

import numpy as np
from numpy.typing import DTypeLike

from .tensor import Tensor, T
from .autograd import functions as func


def empty(
    shape: tuple[int, ...], dtype: DTypeLike = None, *, requires_grad: bool = False
) -> Tensor:
    """
    Create an empty tensor of given shape.
    Args:
        shape (tuple[int, ...]): The shape of the tensor.
        dtype (DTypeLike, optional): The dtype of the tensor.
        requires_grad (bool, optional): Whether the tensor is trainable.
    Returns:
        Tensor: A tensor of given shape.
    """
    return Tensor(np.empty(shape), dtype=dtype, requires_grad=requires_grad)


def zeros_like(
    arr: Tensor[T], dtype: DTypeLike = None, *, requires_grad: bool = False
) -> Tensor[T]:
    """
    Create a tensor filled with zeros with the same shape and dtype as the input tensor.
    Args:
        arr (Tensor): The input tensor.
        dtype (DTypeLike): The data type of the output tensor.
        requires_grad (bool): Flag to enable gradient computation.
    Returns:
        Tensor: A tensor filled with zeros with the same shape and dtype as the input tensor.
    """
    return Tensor(
        np.zeros_like(arr.data, dtype=dtype),
        dtype=dtype,
        requires_grad=requires_grad,
    )


def zeros(
    shape: tuple[int, ...],
    dtype: DTypeLike = None,
    *,
    requires_grad: bool = False,
) -> Tensor[T]:
    """
    Create a tensor filled with zeros.
    Args:
        shape (tuple[int, ...]): The shape of the tensor.
        dtype (DTypeLike): The data type of the output tensor.
        requires_grad (bool): Flag to enable gradient computation.
    Returns:
        Tensor: A tensor filled with zeros.
    """
    return Tensor(
        np.zeros(shape, dtype=dtype),
        dtype=dtype,
        requires_grad=requires_grad,
    )


def exp(input: Tensor[T], *, inplace: bool = False) -> Tensor[T]:
    """
    Compute the exponential of all elements in the input tensor.
    Args:
        input (Tensor): The input tensor.
        inplace (bool): Flag to modify the input tensor.
    Returns:
        Tensor: The exponential of all elements in the input tensor.
    """
    if inplace:
        input.data[:] = np.e**input.data
        return input

    return np.e**input


def sum(
    input: Tensor[T],
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor[T] | T:
    """
    Compute the sum of all elements in the input tensor.
    Args:
        input (Tensor): The input tensor.
        axis (int | tuple[int, ...] | None): The axis along which the sum is computed.
        keepdims (bool): Flag to keep the dimensions of the input tensor.
    Returns:
        Tensor | T: The sum of all elements in the input tensor.
    """
    return input.apply_operation(func.Sum(input, axis=axis, keepdims=keepdims))


def max(
    input: Tensor[T],
    axis: SupportsIndex | Sequence[SupportsIndex] | None = None,
    keepdims: bool = False,
) -> Tensor[T] | T:
    """
    Compute the maximum of all elements in the input tensor.
    Args:
        input (Tensor): The input tensor.
        axis (SupportsIndex | Sequence[SupportsIndex] | None): The axis along which the maximum is computed.
        keepdims (bool): Flag to keep the dimensions of the input tensor.
    Returns:
        Tensor | T: The maximum of all elements in the input tensor.
    """
    return input.apply_operation(func.Max(input, axis=axis, keepdims=keepdims))

def min(
    input: Tensor[T],
    axis: SupportsIndex | Sequence[SupportsIndex] | None = None,
    keepdims: bool = False,
) -> Tensor[T] | T:
    """
    Compute the minimum of all elements in the input tensor.
    Args:
        input (Tensor): The input tensor.
        axis (SupportsIndex | Sequence[SupportsIndex] | None): The axis along which the minimum is computed.
        keepdims (bool): Flag to keep the dimensions of the input tensor.
    Returns:
        Tensor | T: The minimum of all elements in the input tensor.
    """
    return input.apply_operation(func.Min(input, axis=axis, keepdims=keepdims))

def mean(input: Tensor[T], *, axis: int | tuple[int, ...] | None = None) -> Tensor[T] | T:
    """
    Compute the mean of all elements in the input tensor.
    Args:
        input: The input tensor.
        axis: The axis along which the mean is computed.
    Returns:
        Tensor | T: The mean of all elements in the input tensor.
    """
    return input.apply_operation(func.Mean(input, axis=axis))

def tanh(input: Tensor[T], *, inplace: bool = False) -> Tensor[T]:
    """
    Compute the tangent of all elements in the input tensor.
    Args:
        input (Tensor): The input tensor.
        inplace (bool): Flag to modify the input tensor.
    Returns:
        Tensor: The hyperbolic tangent of all elements in the input tensor.
    """
    return input.apply_operation(func.Tanh(input), inplace=inplace)


def log(input: Tensor[T], *, inplace: bool = False) -> Tensor[T]:
    """
    Compute the natural logarithm of all elements in the input tensor.
    Args:
        input (Tensor): The input tensor.
        inplace (bool): Flag to modify the input tensor.
    Returns:
        Tensor: The natural logarithm of all elements in the input tensor.
    """
    return input.apply_operation(func.Log(input), inplace=inplace)


def reshape(input: Tensor[T], shape: tuple[int, ...]) -> Tensor[T]:
    """
    Reshape the input tensor.
    Args:
        input (Tensor): The input tensor.
        shape (tuple[int, ...]): The new shape of the tensor.
    Returns:
        Tensor: The reshaped tensor.
    """
    return input.apply_operation(func.Reshape(input, shape=shape))


def transpose(input: Tensor[T], *, axes: list[int] | tuple[int, ...] | int | None = None) -> Tensor[T]:
    """
    Transpose the input tensor.
    Args:
        input (Tensor): The input tensor.
        axes (list[int] | tuple[int, ...] | None): The new axes of the tensor.
    Returns:
        Tensor: The transposed tensor
    """
    return input.apply_operation(operation=func.Transpose(input, axes=axes))

def flatten(input: Tensor[T]) -> Tensor[T]:
    """
    Flatten the input tensor.
    Args:
        input: The input tensor.
    Returns:
        Tensor: The flattened tensor.
    """
    return input.apply_operation(operation=func.Flatten(input))

def expand_dims(input: Tensor[T], axis: int | list[int] | tuple[int, ...]) -> Tensor[T] | T:
    """
    Expands the input tensor to the specified dimensions.
    Args:
        input: The input tensor.
        axis: The axis along which the expansion is computed.
    Returns:
        Tensor: The expanded tensor.
    """
    return input.apply_operation(operation=func.ExpandDims(input, axis))

def round(input: Tensor[T], decimals: int = 0, *, inplace: bool = False) -> Tensor[T]:
    """
    Round the input tensor to the specified number of decimals.
    Args:
        input (Tensor): The input tensor.
        decimals (int): The number of decimals to round to.
        inplace (bool): Flag to modify the input tensor.
    Returns:
        Tensor: The rounded tensor.
    """
    return input.apply_operation(func.Round(input, decimals=decimals), inplace=inplace)

def pad(
        input: Tensor[T],
        pad_width: int | tuple,
        *,
        value: T = 0
) -> Tensor[T]:
    """
    Pad the input tensor to the specified number of dimensions.
    Args:
        input: The input tensor.
        pad_width: The number of dimensions to pad.
        value: The value to pad.
    Returns:
        Tensor: The padded tensor.
    """
    return input.apply_operation(func.Pad(input, pad_width=pad_width, value=value))

def compose(tensors: list[Tensor[T]] | tuple[Tensor[T], ...]) -> Tensor[T]:
    """
    Compose the tensors.\n
    The created tensor propagates the gradient through the tensors.
    Args:
        *tensors: List of tensors.

    Returns:
        New tensor composed of the tensors.
    """
    # The tensor used to apply operation doesn't care
    return tensors[0].apply_operation(func.Compose(tensors))

def cce(predicted: Tensor[T], expected: Tensor[T]) -> Tensor[T]:
    """
    Compute the categorical cross-entropy loss.
    Args:
        predicted (Tensor): The predicted tensor.
        expected (Tensor): The expected tensor.
    Returns:
        Tensor: The categorical cross-entropy loss.
    """
    return predicted.apply_operation(func.CategoricalCrossentropy(predicted, expected))
