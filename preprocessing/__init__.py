"""
This module provides preprocessing utilities for preparing data before training neural networks.

Functions:
    - min_max_scaler: Scales features to a given range, typically [0, 1]
    - train_test_split: Splits data into training and testing sets
    - one_hot_encode: Converts categorical labels into a one-hot encoded format

Example:
    >>> from preprocessing import min_max_scaler, train_test_split
    >>> # Example data
    >>> data = np.array([[1, 2], [3, 4], [5, 6]])
    >>> # Scale data
    >>> scaled_data = min_max_scaler(data)
    >>> print(scaled_data)
    >>> # Split data
    >>> train_data, test_data = train_test_split(data, test_size=0.2)
    >>> print(train_data)
    >>> print(test_data)
"""

from .preprocessing import min_max_scaler, train_test_split, one_hot_encode

__all__ = ["min_max_scaler", "train_test_split", "one_hot_encode"]
