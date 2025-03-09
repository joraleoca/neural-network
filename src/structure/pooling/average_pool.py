from typing import Any

import numpy as np

from .pool import Pool
from src.tensor import Tensor, op


class AveragePool(Pool):
    def __call__(self, data: Tensor[np.floating]) -> Tensor[np.floating]:
        data = self._pad(data, const_val=data.dtype.type(0))

        windows = self._windows(data)

        return op.mean(windows, axis=(-1, -2))

    @staticmethod
    def from_data(data: dict[str, Any]) -> "AveragePool":
        return AveragePool(
            channels=data["channels"].item(),
            filter_shape=tuple(data["filter_shape"]),
            stride=data["stride"].item(),
            padding=data["padding"].item()
        )