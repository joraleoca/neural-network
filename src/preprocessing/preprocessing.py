import random

import numpy as np
from numpy.typing import NDArray

from ..constants import EPSILON
from src.core import Tensor


def min_max_scaler(
    data: Tensor[np.floating] | NDArray[np.floating], min_val: float, max_val: float,
) -> Tensor[np.floating]:
    """
    Scales the input data to a specified range [min, max] using min-max normalization.
    Parameters:
        data (Tensor[np.floating] | NDArray[np.floating]): The input data to be scaled. It should be a NumPy array of floating-point numbers.
        min_val (float): The minimum value of the desired range.
        max_val (float): The maximum value of the desired range.
    Returns:
        Tensor[np.floating]: The scaled data with values in the range [min, max].
    Raises:
        ValueError if min is greater than max
    """
    if min_val > max_val:
        raise ValueError(f"{min_val} is greater than {max_val}")

    data_std = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + EPSILON)
    return Tensor(data_std * (max_val - min_val) + min_val)

def train_test_split(
    data: Tensor | NDArray,
    expected: Tensor | NDArray,
    *,
    train_size: int | float | None = None,
    test_size: int | float | None = None,
    shuffle: bool = True,
    random_state = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Split the data into random train and test subsets.

    Args:
        data (Tensor): The data to be split.
        expected (Tensor): The expected output corresponding to the data.
        train_size (int, float, or None, optional): 
            If int, represents the absolute number of train samples.
            If float, represents the proportion of the dataset to include in the train split.
            If None, the value is set to 0.75. Default is None.
        test_size (int, float, or None, optional): 
            If int, represents the absolute number of test samples.
            If float, represents the proportion of the dataset to include in the test split.
            If None, the value is set to the complement of train_size. Default is None.
        random_state (int, optional): Controls the shuffling applied to the data before applying the split. Default is None.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]: The split data (data_train, expected_train, data_test, expected_test).

    ValueError: If `train_size` and `test_size` do not sum up to the number of samples in `data`.
    TypeError: If `train_size` or `test_size` are not int, float, or None.
    """
    data = list(zip(data, expected))

    n_samples = len(data)

    if train_size is None and test_size is None:
        train_size = 0.75

    if isinstance(train_size, float):
        train_count = int(train_size * n_samples)
    elif isinstance(train_size, int):
        train_count = train_size
    elif train_size is None:
        train_count = n_samples - (
            int(test_size * n_samples) if isinstance(test_size, float) else test_size
        )
    else:
        raise TypeError("train_size must be int, float, or None")

    if isinstance(test_size, float):
        test_count = int(test_size * n_samples)
    elif isinstance(test_size, int):
        test_count = test_size
    elif test_size is None:
        test_count = n_samples - train_count
    else:
        raise TypeError("test_size must be int, float, or None")

    if train_count + test_count > n_samples:
        raise ValueError(
            f"train_size({train_count}) + test_size({test_count}) > n_samples({n_samples})"
        )

    if shuffle:
        random.Random(random_state).shuffle(data)

    train_data = data[:train_count]
    test_data = data[train_count : train_count + test_count]

    data_train, expected_train = zip(*train_data)
    data_test, expected_test = zip(*test_data)

    return data_train, expected_train, data_test, expected_test
