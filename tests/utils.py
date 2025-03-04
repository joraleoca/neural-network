import numpy as np
from numpy.typing import NDArray

from src.core import Tensor


def assert_grad(arr: Tensor, grad: NDArray, err: str | None = None) -> None:
    """
    Asserts that the gradient of a given tensor matches the expected gradient.
    Args:
        arr (Tensor): The tensor whose gradient is to be checked.
        grad (NDArray): The expected gradient array.
        err (str): The error message to display if the gradient does not match the expected gradient.
    Raises:
        AssertionError: If the gradient of the tensor does not match the expected gradient.
    """
    if err is None:
        err = f"Gradient of {arr} should be {grad}. Got {arr.grad}"

    assert np.allclose(arr.grad, grad), err


def assert_data(arr: Tensor, data: Tensor | NDArray, err: str | None = None) -> None:
    """
    Asserts that the data of a given tensor matches the expected data.
    Args:
        arr (Tensor): The tensor whose data is to be checked.
        data (Tensor | NDArray): The expected data to compare against the tensor's data.
        err (str): The error message to display if the tensor's data does not match the expected data.
    Raises:
        AssertionError: If the tensor's data does not match the expected data.
    """
    if err is None:
        err = f"Data of {arr} should be {data}. Got {arr.data}"

    assert np.allclose(arr, data), err
