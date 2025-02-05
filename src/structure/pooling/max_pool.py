from typing import Any

import numpy as np

from .pool import Pool
from src.core import Tensor, op


class MaxPool(Pool):
    def forward(self, data: Tensor[np.floating]) -> Tensor[np.floating]:
        data = self._pad(data, const_val=data.dtype.type(float("-inf")))

        output_height, output_width = self._output_dimensions(data.shape[-2:])
        batch_size = data.shape[0]

        windows = op.compose([
            data[
                :,
                :,
                i : i + self.filter_size[1],
                j : j + self.filter_size[0],
            ]
            for i in range(0, output_height * self.stride, self.stride)
            for j in range(0, output_width * self.stride, self.stride)
        ])

        windows = windows.reshape(
            (batch_size, output_height, output_width, self.channels, self.filter_size[1], self.filter_size[0])
        )

        return op.transpose(op.max(windows, axis=(-1, -2)), axes=(0, 3, 1, 2))

    @staticmethod
    def from_data(data: dict[str, Any]) -> "MaxPool":
        return MaxPool(
            channels=data["channels"].item(),
            filter_size=tuple(data["filter_size"]), #type: ignore
            stride=data["stride"].item(),
            padding=data["padding"].item(),
        )
