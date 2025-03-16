from typing import Callable, Sequence, SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from .autograd import functions as func
from .autograd import operations as ops
from .tensor import T, Tensor


def ensure_input_tensor(func: Callable) -> Callable:
    """Decorator to ensure the input is a Tensor."""

    def wrapper(input, *args, **kwargs) -> Callable:
        if not isinstance(input, Tensor):
            input = Tensor(input)
        return func(input, *args, **kwargs)

    return wrapper


def empty(shape: tuple[int, ...], dtype: DTypeLike = None, *, requires_grad: bool = False) -> Tensor:
    """
    Create an empty tensor of given shape.
    Args:
        shape (tuple[int, ...]): The shape of the tensor.
        dtype (DTypeLike, optional): The dtype of the tensor.
        requires_grad (bool, optional): Whether the tensor is trainable.
    Returns:
        Tensor: A tensor of given shape.
    """
    return Tensor(np.empty(shape), dtype=dtype or Tensor.default_dtype, requires_grad=requires_grad)


def zeros_like(arr: Tensor[T], dtype: DTypeLike = None, *, requires_grad: bool = False) -> Tensor[T]:
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
        np.zeros_like(arr.data),
        dtype=dtype or arr.dtype,
        requires_grad=requires_grad,
    )


def zeros(
    shape: tuple[int, ...],
    dtype: DTypeLike = None,
    *,
    requires_grad: bool = False,
) -> Tensor:
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
        np.zeros(shape),
        dtype=dtype or Tensor.default_dtype,
        requires_grad=requires_grad,
    )


def ones(shape: tuple[int, ...], dtype: DTypeLike = None, *, requires_grad: bool = False) -> Tensor:
    """
    Create a tensor filled with ones.
    Args:
        shape (tuple[int, ...]): The shape of the tensor.
        dtype (DTypeLike): The data type of the output tensor.
        requires_grad (bool): Flag to enable gradient computation.
    Returns:
        Tensor: A tensor filled with ones.
    """
    return Tensor(
        np.ones(shape),
        dtype=dtype or Tensor.default_dtype,
        requires_grad=requires_grad,
    )


@ensure_input_tensor
def exp(input: Tensor[T] | ArrayLike | T, *, inplace: bool = False) -> Tensor[T]:
    """
    Compute the exponential of all elements in the input.
    Args:
        input (Tensor): The input data.
        inplace (bool): Flag to modify the input tensor.
    Returns:
        Tensor: The exponential of all elements in the input tensor.
    """
    return ops.Pow.forward(Tensor(np.e, dtype=Tensor.default_dtype), input, inplace=inplace)


@ensure_input_tensor
def sqrt(input: Tensor[T] | ArrayLike | T, *, inplace: bool = False) -> Tensor[T]:
    """
    Compute the square root of all elements in the input.
    Args:
        input (Tensor): The input data.
        inplace (bool): Flag to modify the input tensor.
    Returns:
        Tensor: The square root of all elements in the input tensor.
    """
    return ops.Sqrt.forward(input, inplace=inplace)


@ensure_input_tensor
def sum(
    input: Tensor[T] | ArrayLike | T,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor[T]:
    """
    Compute the sum of all elements in the input.
    Args:
        input (Tensor | ArrayLike | T): The input data.
        axis (int | tuple[int, ...] | None): The axis along which the sum is computed.
        keepdims (bool): Flag to keep the dimensions of the input tensor.
    Returns:
        Tensor: The sum of all elements in the input.
    """
    return func.Sum.forward(input, axis=axis, keepdims=keepdims)


@ensure_input_tensor
def max(
    input: Tensor[T] | ArrayLike | T,
    axis: SupportsIndex | Sequence[SupportsIndex] | None = None,
    keepdims: bool = False,
) -> Tensor[T]:
    """
    Compute the maximum of all elements in the input.
    Args:
        input (Tensor | ArrayLike | T): The input data.
        axis (SupportsIndex | Sequence[SupportsIndex] | None): The axis along which the maximum is computed.
        keepdims (bool): Flag to keep the dimensions of the input.
    Returns:
        Tensor: The maximum of all elements in the input.
    """
    return func.Max.forward(input, axis=axis, keepdims=keepdims)


@ensure_input_tensor
def min(
    input: Tensor[T] | ArrayLike | T,
    axis: SupportsIndex | Sequence[SupportsIndex] | None = None,
    keepdims: bool = False,
) -> Tensor[T]:
    """
    Compute the minimum of all elements in the input.
    Args:
        input (Tensor | ArrayLike | T): The input data.
        axis (SupportsIndex | Sequence[SupportsIndex] | None): The axis along which the minimum is computed.
        keepdims (bool): Flag to keep the dimensions of the input tensor.
    Returns:
        Tensor: The minimum of all elements in the input tensor.
    """
    return func.Min.forward(input, axis=axis, keepdims=keepdims)


@ensure_input_tensor
def mean(
    input: Tensor[T] | ArrayLike | T, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Tensor[T]:
    """
    Compute the mean of all elements in the input.
    Args:
        input (Tensor | ArrayLike | T): The input data.
        axis (int | tuple[int, ...]): The axis along which the mean is computed.
        keepdims (bool): Flag to keep the dimensions of the input tensor.
    Returns:
        Tensor: The mean of all elements in the input.
    """
    return func.Mean.forward(input, axis=axis, keepdims=keepdims)


@ensure_input_tensor
def tanh(input: Tensor[T] | ArrayLike | T, *, inplace: bool = False) -> Tensor[T]:
    """
    Compute the tangent of all elements in the input.
    Args:
        input (Tensor | ArrayLike | T): The input data.
        inplace (bool): Flag to modify the input.
    Returns:
        Tensor: The hyperbolic tangent of all elements in the input.
    """
    return func.Tanh.forward(input, inplace=inplace)


@ensure_input_tensor
def log(input: Tensor[T] | ArrayLike | T, *, inplace: bool = False) -> Tensor[T]:
    """
    Compute the natural logarithm of all elements in the input.
    Args:
        input (Tensor): The input data.
        inplace (bool): Flag to modify the input.
    Returns:
        Tensor: The natural logarithm of all elements in the input.
    """
    return func.Log.forward(input, inplace=inplace)


def reshape(input: Tensor[T], shape: tuple[int, ...], *, inplace: bool = False) -> Tensor[T]:
    """
    Reshape the input tensor.
    Args:
        input (Tensor): The input tensor.
        shape (tuple[int, ...]): The new shape of the tensor.
    Returns:
        Tensor: The reshaped tensor.
    """
    return func.Reshape.forward(input, shape=shape, inplace=inplace)


def transpose(input: Tensor[T], *, axes: list[int] | tuple[int, ...] | int | None = None) -> Tensor[T]:
    """
    Transpose the input tensor.
    Args:
        input (Tensor): The input tensor.
        axes (list[int] | tuple[int, ...] | None): The new axes of the tensor.
    Returns:
        Tensor: The transposed tensor
    """
    return func.Transpose.forward(input, axes=axes)


def flatten(input: Tensor[T]) -> Tensor[T]:
    """
    Flatten the input tensor.
    Args:
        input: The input tensor.
    Returns:
        Tensor: The flattened tensor.
    """
    return func.Flatten.forward(input)


def expand_dims(input: Tensor[T], axis: int | list[int] | tuple[int, ...]) -> Tensor[T]:
    """
    Expands the input tensor to the specified dimensions.
    Args:
        input: The input tensor.
        axis: The axis along which the expansion is computed.
    Returns:
        Tensor: The expanded tensor.
    """
    return func.ExpandDims.forward(input, axis)


@ensure_input_tensor
def round(input: Tensor[T] | ArrayLike | T, decimals: int = 0, *, inplace: bool = False) -> Tensor[T]:
    """
    Round the input to the specified number of decimals.
    Args:
        input (Tensor | ArrayLike | T): The input data.
        decimals (int): The number of decimals to round to.
        inplace (bool): Flag to modify the input.
    Returns:
        Tensor: The rounded tensor.
    """
    return func.Round.forward(input, decimals=decimals, inplace=inplace)


@ensure_input_tensor
def pad(input: Tensor[T] | ArrayLike, pad_width: int | tuple, *, value: T = 0) -> Tensor[T]:
    """
    Pad the input tensor to the specified number of dimensions.
    Args:
        input: The input tensor.
        pad_width: The number of dimensions to pad.
        value: The value to pad.
    Returns:
        Tensor: The padded tensor.
    """
    return func.Pad.forward(input, pad_width=pad_width, value=value)


def stack(tensors: list[Tensor[T]] | tuple[Tensor[T], ...], axis: int = 0) -> Tensor[T]:
    """
    Stack the tensors.\n
    The created tensor propagates the gradient through the tensors.
    Args:
        *tensors: List of tensors.

    Returns:
        New tensor composed of the tensors.
    """
    return func.Stack.forward(*tensors, axis=axis)


def cce(predicted: Tensor[T], expected: Tensor[T]) -> Tensor[T]:
    """
    Compute the categorical cross-entropy loss.
    Args:
        predicted (Tensor): The predicted tensor.
        expected (Tensor): The expected tensor.
    Returns:
        Tensor: The categorical cross-entropy loss.
    """
    return func.CategoricalCrossentropy.forward(predicted, expected)


@ensure_input_tensor
def as_strided(input: Tensor[T] | ArrayLike | T, *, shape: tuple[int, ...], strides: tuple[int, ...]) -> Tensor[T]:
    """
    Create a strided tensor from the input.
    Args:
        input (Tensor | ArrayLike | T): The input data.
        shape (tuple[int, ...]): The shape of the tensor.
        strides (tuple[int, ...]): The strides of the tensor.
    Returns:
        Tensor: The strided tensor.
    """
    return func.As_Strided.forward(input, shape=shape, strides=strides)


@ensure_input_tensor
def argmax(input: Tensor[T] | ArrayLike | T, *, axis: SupportsIndex | None = None, keepdims: bool = False) -> Tensor[T]:
    """
    Compute the indices of the maximum values along the specified axis.
    Args:
        input (Tensor | ArrayLike | T): The input data.
        axis (SupportsIndex | None): The axis along which the maximum is computed.
        keepdims (bool): Flag to keep the dimensions of the input tensor.
    Returns:
        Tensor: The indices of the maximum values along the specified axis.
    """
    return func.Argmax.forward(input, axis=axis, keepdims=keepdims)
