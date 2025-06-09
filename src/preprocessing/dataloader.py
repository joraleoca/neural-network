from typing import Self, Callable

import numpy as np
from numpy.typing import NDArray

from src.tensor import Tensor


class DataLoader[T: Tensor | NDArray | list]:
    __slots__ = "data", "expected", "preprocess_data", "preprocess_expected", "batch_size", "shuffle", "_index", "_p"

    def __init__(
        self,
        data: T,
        expected: T,
        preprocess_data: Callable[[T], T] | None = None,
        preprocess_expected: Callable[[T], T] | None = None,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> None:
        """
        DataLoader class to load data in batches.

        Args:
            data (T): The data to load.
            expected (T): The expected values.
            preprocess_data (Callable[[T], T]): Function to preprocess the data.
            preprocess_expected (Callable[[T], T]): Function to preprocess the expected values.
            batch_size (int): The batch size.
        """
        if batch_size <= 0:
            raise ValueError(f"batch size must be positive. Got {batch_size}")

        if len(data) != len(expected):
            raise ValueError(
                f"Data length different than expected length. Got data lenght: {len(data)}, expected length: {len(expected)}"
            )
        if len(data) < batch_size:
            raise ValueError(
                f"The data length is less than the batch size. Got data length: {len(data)}, batch size: {batch_size}"
            )

        self.data = data
        self.expected = expected
        self.preprocess_data = preprocess_data
        self.preprocess_expected = preprocess_expected
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.__iter__()  # Makes the object iterable from default

    def __iter__(self) -> Self:
        if self.shuffle:
            self._p = np.random.permutation(len(self.data))
        self._index = 0
        return self

    def __next__(self) -> tuple[Tensor | list, Tensor | list]:
        if self._index + self.batch_size >= len(self.data):
            raise StopIteration("No more data to load.")

        batch = slice(self._index, self._index + self.batch_size)
        self._index += self.batch_size

        data, expected = None, None
        if self.shuffle:
            batch = self._p[batch]

            if isinstance(self.data, list):
                data = [self.data[idx] for idx in batch.ravel()]
            if isinstance(self.expected, list):
                expected = [self.expected[idx] for idx in batch.ravel()]

        if data is None:
            data = self.data[batch]
        if expected is None:
            expected = self.expected[batch]

        if self.preprocess_data is not None:
            data = self.preprocess_data(data)
        if self.preprocess_expected is not None:
            expected = self.preprocess_expected(expected)

        return data, expected
