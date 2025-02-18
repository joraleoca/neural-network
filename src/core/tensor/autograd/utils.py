from copy import deepcopy

import cupy as cp
from numpy.typing import NDArray

from src.core.tensor import tensor as t

def update_broadcast_grad(grad: NDArray, tensor: "t.Tensor") -> NDArray:
    """
    Computes the broadcast gradient with respect to `respect_to`.
    Args:
        grad: The gradient.
        tensor: The tensor whose broadcast gradient is to be computed.

    Returns:
        The updated gradient.
    """
    xp = cp.get_array_module(grad)
    if tensor.size == 1:
        return xp.atleast_1d(grad.sum())

    shape = list(grad.shape)
    tensor_shape = [0] * max(0, len(shape) - len(tensor.shape)) + list(tensor.shape)

    shapes_news = []
    shapes_ones = []

    for i, (s, r_s) in enumerate(zip(shape, tensor_shape)):
        diff = r_s - s

        if diff < 0:
            shapes_news.append(i)
        elif diff > 0 and tensor_shape[i] != 1:
            shapes_ones.append(i)

    grad = grad.sum(tuple(shapes_ones), keepdims=True)
    grad = grad.sum(tuple(shapes_news))

    return xp.atleast_1d(grad)


def update_tensor_grad(tensor: "t.Tensor", grad: NDArray) -> None:
    """
    Stores the gradient of `tensor`, fixing its shape and broadcasting.
    Args:
        tensor: The tensor whose gradient is to be updated.
        grad: The updated gradient.
    """
    gr = update_broadcast_grad(grad, tensor)

    if tensor.grad is None:
        tensor.grad = deepcopy(gr)
    else:
        tensor.grad += gr
