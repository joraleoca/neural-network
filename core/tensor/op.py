import numpy as np
from numpy.typing import DTypeLike

from .tensor import Tensor, T
from .autograd import functions as func


def zeros_like(
    input: Tensor[T], dtype: DTypeLike = None, *, requires_grad: bool = False
) -> Tensor[T]:
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
    return Tensor(
        np.zeros(shape, dtype=dtype),
        dtype=dtype,
        requires_grad=requires_grad,
    )


def exp(input: Tensor[T], *, inplace: bool = False) -> Tensor[T]:
    if inplace:
        input.data[:] = np.e**input.data  # type: ignore
        return input

    return np.e**input


def sum(
    input: Tensor[T],
    axis: int | None = None,
    keepdims: bool = False,
) -> Tensor[T] | T:
    return input.apply_operation(func.Sum(input, axis=axis, keepdims=keepdims))


def max(
    input: Tensor[T],
    axis: int | None = None,
    keepdims: bool = False,
) -> Tensor[T] | T:
    return input.apply_operation(func.Max(input, axis=axis, keepdims=keepdims))


def tanh(input: Tensor[T], *, inplace: bool = False) -> Tensor[T]:
    return input.apply_operation(func.Tanh(input), inplace=inplace)


def log(input: Tensor[T], *, inplace: bool = False) -> Tensor[T]:
    return input.apply_operation(func.Log(input), inplace=inplace)


def reshape(input: Tensor[T], shape: tuple[int, ...]) -> Tensor[T]:
    return input.apply_operation(func.Reshape(input, shape=shape))


def transpose(input: Tensor[T]) -> Tensor[T]:
    return input.apply_operation(operation=func.Transpose(input))


def round(input: Tensor[T], decimals: int = 0, *, inplace: bool = False) -> Tensor[T]:
    return input.apply_operation(func.Round(input, decimals=decimals), inplace=inplace)


def cce(predicted: Tensor[T], expected: Tensor[T]) -> Tensor[T]:
    return predicted.apply_operation(func.CategoricalCrossentropy(predicted, expected))
