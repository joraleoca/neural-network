import numpy as np
from numpy.typing import NDArray

from core import Tensor


def assert_grad(arr: Tensor, grad: NDArray) -> None:
    """
    Asserts that the gradient of a given tensor matches the expected gradient.
    Args:
        arr (Tensor): The tensor whose gradient is to be checked.
        grad (NDArray): The expected gradient array.
    Raises:
        AssertionError: If the gradient of the tensor does not match the expected gradient.
    """
    assert np.allclose(arr.grad, grad), f"'{arr.grad}' should be {grad}"


def assert_data(arr: Tensor, data: NDArray) -> None:
    """
    Asserts that the data of a given tensor matches the expected data.
    Args:
        arr (Tensor): The tensor whose data is to be checked.
        data (NDArray): The expected data to compare against the tensor's data.
    Raises:
        AssertionError: If the tensor's data does not match the expected data.
    """
    assert np.allclose(arr.data, data), f"'{arr}' data should be {data}"
