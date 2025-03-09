from typing import Self

import cupy as cp
from numpy.typing import ArrayLike

from src.tensor import Tensor


class DataLoader:
    __slots__ = "data", "expected", "batch_size", "shuffle", "_index", "_p"

    def __init__(
        self, data: Tensor | ArrayLike, expected: Tensor | ArrayLike, batch_size: int = 32, shuffle: bool = True
    ) -> None:
        """
        DataLoader class to load data in batches.

        Args:
            data (Tensor | ArrayLike): The data to load.
            expected (Tensor | ArrayLike): The expected values.
            batch_size (int): The batch size.
        """
        if batch_size <= 0:
            raise ValueError(f"batch size must be positive. Got {batch_size}")

        if not isinstance(data, Tensor):
            data = Tensor(data)
        if not isinstance(expected, Tensor):
            expected = Tensor(expected)

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
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.__iter__()  # Makes the object iterable from default

    def __iter__(self) -> Self:
        if self.shuffle:
            xp = cp.get_array_module(self.data.data)
            self._p = xp.random.permutation(len(self.data))
        self._index = 0
        return self

    def __next__(self) -> tuple[Tensor, Tensor]:
        if self._index + self.batch_size >= len(self.data):
            if self.shuffle:
                xp = cp.get_array_module(self.data.data)
                xp.random.shuffle(self._p)
            self._index = 0

        batch = slice(self._index, self._index + self.batch_size)
        self._index += self.batch_size

        if self.shuffle:
            batch = self._p[batch]

        return self.data[batch], self.expected[batch]
