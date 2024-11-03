from typing import Any

import numpy as np
from numpy.typing import NDArray


def min_max_scaler(
    data: NDArray[np.floating[Any]], min: float, max: float
) -> NDArray[np.floating[Any]]:
    """
    Scales the input data to a specified range [min, max] using min-max normalization.
    Parameters:
        data (NDArray[np.floating[Any]]): The input data to be scaled. It should be a NumPy array of floating-point numbers.
        min (float): The minimum value of the desired range.
        max (float): The maximum value of the desired range.
    Returns:
        NDArray[np.floating[Any]]: The scaled data with values in the range [min, max].
    Raises:
        ValueError if min is greater than max
    """
    if min > max:
        raise ValueError(f"{min} is greater than {max}")

    data_std = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    return data_std * (max - min) + min


def train_test_split(
    data: NDArray,
    *,
    train_size: int | float | None = None,
    test_size: int | float | None = None,
    random_state=None,
    shuffle: bool = True,
) -> tuple[NDArray, NDArray]:
    """
    Splits the data into training and testing sets.
    Parameters:
        data (NDArray): The data to be split.
        train_size (int, float, or None):
            If int, represents the absolute number of train samples.\n
            If float, represents the proportion of the dataset to include in the train split.\n
            If None, the value is set to 0.75.
        test_size (int, float, or None):
            If int, represents the absolute number of test samples.\n
            If float, represents the proportion of the dataset to include in the test split.\n
            If None, the value is set to the complement of train_size.
        random_state: Controls the shuffling applied to the data before applying the split.
        shuffle (bool, optional): Whether or not to shuffle the data before splitting. Default is True.
    Returns:
        tuple[NDArray, NDArray]: A tuple containing the training data and the testing data in that order.
    Raises:
        ValueError:
            If `train_size` and `test_size` do not sum up to the number of samples in `data`.
        TypeError:
            If `train_size` or `test_size` are not int, float, or None.
    """
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
        )  # type: ignore
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
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    return data[indices[:train_count]], data[
        indices[train_count : train_count + test_count]
    ]
