from typing import Any

import numpy as np

from .pool import Pool
from src.core import Tensor, op


class MaxPool(Pool):
    def forward(self, data: Tensor[np.floating]) -> Tensor[np.floating]:
        data = self._pad(data, const_val=data.dtype.type(float("-inf")))

        windows = self._windows(data)

        return op.max(windows, axis=(-1, -2))

    @staticmethod
    def from_data(data: dict[str, Any]) -> "MaxPool":
        return MaxPool(
            channels=data["channels"].item(),
            filter_size=tuple(data["filter_size"]),
            stride=data["stride"].item(),
            padding=data["padding"].item(),
        )
