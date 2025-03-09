from copy import deepcopy

import cupy as cp
from numpy.typing import NDArray

from .. import tensor as t


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

    grad_shape = grad.shape
    tensor_shape = tensor.shape

    dim_diff = len(grad_shape) - len(tensor_shape)
    if dim_diff > 0:
        grad = grad.sum(axis=tuple(range(dim_diff)))

    axes_to_sum = tuple(i for i, t in enumerate(tensor_shape) if t == 1)
    if axes_to_sum:
        grad = grad.sum(axis=axes_to_sum, keepdims=True)

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
