from typing import Callable, Sequence, SupportsIndex, Any

import cupy as cp
import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from .autograd import functions as func
from .autograd import operations as ops
from .tensor import Tensor, T


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
    return Tensor(Tensor.default_module.empty(shape), dtype=dtype or Tensor.default_dtype, requires_grad=requires_grad)


def zeros_like(arr: Tensor, dtype: DTypeLike = None, *, requires_grad: bool = False) -> Tensor:
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
        Tensor.default_module.zeros_like(arr.data),
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
        Tensor.default_module.zeros(shape),
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
        Tensor.default_module.ones(shape),
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


def stack(
    tensors: list[Tensor[T]] | tuple[Tensor[T], ...], axis: int = 0, *, dtype: str | DTypeLike = None
) -> Tensor[T]:
    """
    Stack the tensors.\n
    The created tensor propagates the gradient through the tensors.
    Args:
        *tensors: List of tensors.

    Returns:
        New tensor composed of the tensors.
    """
    return func.Stack.forward(*tensors, axis=axis, dtype=dtype)


def concat(
    tensors: list[Tensor[T]] | tuple[Tensor[T], ...], axis: int = 0, *, dtype: str | DTypeLike = None
) -> Tensor[T]:
    """
    Concatenate the tensors along the specified axis.
    Args:
        tensors (list[Tensor[T]] | tuple[Tensor[T], ...]): The tensors to concatenate.
        axis (int): The axis along which to concatenate.
        dtype (str | DTypeLike, optional): The data type of the output tensor.
    Returns:
        Tensor: The concatenated tensor.
    """
    return func.Concat.forward(*tensors, axis=axis, dtype=dtype)


def cce(predicted: Tensor[T], expected: Tensor[T], ignore_token_id: int | None = None) -> Tensor[T]:
    """
    Compute the categorical cross-entropy loss.
    Args:
        predicted (Tensor): The predicted tensor.
        expected (Tensor): The expected tensor.
        ignore_token_id (int | None, optional): The index to ignore in the loss computation.
    Returns:
        Tensor: The categorical cross-entropy loss.
    """
    return func.CategoricalCrossentropy.forward(predicted, expected, ignore_token_id=ignore_token_id)


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


@ensure_input_tensor
def repeat(input: Tensor[T] | ArrayLike | T, repeats: int, axis: SupportsIndex | None = None) -> Tensor[T]:
    """
    Repeat the input tensor.
    Args:
        input (Tensor): The input tensor.
        repeats (int): The number of repetitions.
        axis (SupportsIndex | None): The axis along which the repetition is computed.
    Returns:
        Tensor: The repeated tensor.
    Notes:
        Backward computation is not supported.
    """
    xp = cp.get_array_module(input.data)
    return Tensor(xp.repeat(input.data, repeats, axis=axis), dtype=input.dtype)


@ensure_input_tensor
def triu(input: Tensor[T] | ArrayLike | T, k: int = 0) -> Tensor[T]:
    """
    Return the upper triangular part of the input tensor.
    Args:
        input (Tensor | ArrayLike | T): The input data.
        k (int): The diagonal above which to zero elements.
    Returns:
        Tensor: The upper triangular part of the input tensor.
    """
    return func.Triu.forward(input, k=k)


@ensure_input_tensor
def relu(input: Tensor[T] | ArrayLike | T) -> Tensor[T]:
    return input * (input > 0)


@ensure_input_tensor
def leaky_relu(input: Tensor[T] | ArrayLike | T, alpha: float) -> Tensor[T]:
    if alpha < 0:
        raise ValueError(f"Alpha must be non-negative. Got {alpha}.")

    xp = cp.get_array_module(input.data)
    return input * xp.where(input > 0, xp.ones((1,), dtype=input.dtype), xp.array([alpha], dtype=input.dtype))


@ensure_input_tensor
def sigmoid(input: Tensor[T] | ArrayLike | T) -> Tensor[T]:
    return 1 / (1 + exp(-input))


@ensure_input_tensor
def softmax(input: Tensor[T] | ArrayLike | T) -> Tensor[T]:
    exp_shifted = exp(input - max(input, axis=-1, keepdims=True))
    softmax = exp_shifted / sum(exp_shifted, axis=-1, keepdims=True)

    return softmax


@ensure_input_tensor
def dropout(input: Tensor[T] | ArrayLike | T, p: float, *, rng: Any = None) -> Tensor[T]:
    if not (0 <= p <= 1):
        raise ValueError("The dropout probability must be between 0 and 1.")

    if p == 0:
        return input

    xp = cp.get_array_module(input.data)

    mask = Tensor(xp.random.default_rng(rng).binomial(1, 1 - p, size=input.shape) / (1 - p), dtype=input.dtype)

    return input * mask


def where(condition: Tensor[np.bool], x: Tensor[T], y: Tensor[T]) -> Tensor[T]:
    """
    Select elements from `x` or `y` based on the condition.
    Args:
        condition (Tensor[np.bool]): A boolean tensor that determines which elements to select.
        x (Tensor[T]): The tensor from which to select elements when the condition is True.
        y (Tensor[T]): The tensor from which to select elements when the condition is False.
    Returns:
        Tensor[T]: A tensor containing elements from `x` where the condition is True, and from `y` where the condition is False.
    """
    return func.Where.forward(condition, x, y)


def dotproduct_attention(
    queries: Tensor[T],
    keys: Tensor[T],
    values: Tensor[T],
    attn_mask: Tensor[np.bool] | None = None,
    dropout_p: float = 0,
    *,
    rng: Any = None,
) -> Tensor[T]:
    """
    Computes the dot-product attention.

    Args:
        queries (Tensor[T]): The query tensor of shape (batch size, num queries, d).
        keys (Tensor[T]): The key tensor of shape (batch size, num key-value pairs, d).
        values (Tensor[T]): The value tensor of shape (batch size, num key-value pairs, d).
        attn_mask (Tensor[bool], optional): The valid lengths tensor as (batch size,) or (batch size, num queries).
        dropout_p (float, optional): The probability of dropping out elements. Default is 0.
        rng (Any, optional): Random number generator for dropout. Default is None.

    Returns:
        Tensor[T]: The result of the attention mechanism applied to the values tensor.
    """
    scores: Tensor = queries @ transpose(keys, axes=(*range(0, keys.ndim - 2), -1, -2)) / sqrt(queries.shape[-1])

    if attn_mask is not None:
        if attn_mask.ndim == 2:
            attn_mask = attn_mask[:, None, :]

        scores += attn_mask

    weights = softmax(scores)

    if Tensor.training:
        weights = dropout(weights, dropout_p, rng=rng)

    return weights @ values
