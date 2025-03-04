from typing import Any

import numpy as np

from .pool import Pool
from src.core import Tensor, op
from src.constants import EPSILON


class MaxPool(Pool):
    def forward(self, data: Tensor[np.floating]) -> Tensor[np.floating]:
        data = self._pad(data, const_val=data.dtype.type(-EPSILON))

        windows = self._windows(data)

        return op.max(windows, axis=(-1, -2))

    @staticmethod
    def from_data(data: dict[str, Any]) -> "MaxPool":
        return MaxPool(
            channels=data["channels"].item(),
            filter_shape=tuple(data["filter_shape"]),
            stride=data["stride"].item(),
            padding=data["padding"].item(),
        )
