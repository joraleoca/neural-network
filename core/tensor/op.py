import numpy as np
from numpy.typing import DTypeLike

from .tensor import Tensor, T
from .autograd import functions as func


def zeros_like(
    input: Tensor[T], dtype: DTypeLike = None, *, requires_grad: bool = False
) -> Tensor[T]:
    """
    Create a tensor filled with zeros with the same shape and dtype as the input tensor.

    Args:
        input (Tensor): The input tensor.
        dtype (DTypeLike): The data type of the output tensor.
        requires_grad (bool): Flag to enable gradient computation.

    Returns:
        Tensor: A tensor filled with zeros with the same shape and dtype as the input tensor.
    """
    return Tensor(
        np.zeros_like(input.data, dtype=dtype),
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
        input.data[:] = np.e**input.data  # type: ignore
        return input

    return np.e**input


def sum(
    input: Tensor[T],
    axis: int | None = None,
    keepdims: bool = False,
) -> Tensor[T] | T:
    """
    Compute the sum of all elements in the input tensor.

    Args:
        input (Tensor): The input tensor.
        axis (int): The axis along which the sum is computed.
        keepdims (bool): Flag to keep the dimensions of the input tensor.

    Returns:
        Tensor | T: The sum of all elements in the input tensor.
    """
    return input.apply_operation(func.Sum(input, axis=axis, keepdims=keepdims))


def max(
    input: Tensor[T],
    axis: int | None = None,
    keepdims: bool = False,
) -> Tensor[T] | T:
    """
    Compute the maximum of all elements in the input tensor.

    Args:
        input (Tensor): The input tensor.
        axis (int): The axis along which the maximum is computed.
        keepdims (bool): Flag to keep the dimensions of the input tensor.

    Returns:
        Tensor | T: The maximum of all elements in the input tensor.
    """
    return input.apply_operation(func.Max(input, axis=axis, keepdims=keepdims))


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


def transpose(input: Tensor[T]) -> Tensor[T]:
    """
    Transpose the input tensor.

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor: The transposed tensor
    """
    return input.apply_operation(operation=func.Transpose(input))


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
