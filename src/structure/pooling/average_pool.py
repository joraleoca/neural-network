from typing import Any

import numpy as np

from .pool import Pool
from src.core import Tensor, op


class AveragePool(Pool):
    def forward(self, data: Tensor[np.floating]) -> Tensor[np.floating]:
        # Padding not applied in the channels dimensions
        data = self._pad(data, const_val=data.dtype.type(0))

        output_height, output_width = self._output_dimensions(data.shape[-2:])

        windows = op.compose([
            data[
                :,  # Extract all the input channels
                i : i + self.filter_size[1],
                j : j + self.filter_size[0],
            ]
            for i in range(0, output_height * self.stride, self.stride)
            for j in range(0, output_width * self.stride, self.stride)
        ])

        windows = windows.reshape(
            (output_height, output_width, self.channels, self.filter_size[1], self.filter_size[0])
        )

        return op.transpose(op.mean(windows, axis=(-1, -2)), axes=(2, 0, 1))

    @staticmethod
    def from_data(data: dict[str, Any]) -> "AveragePool":
        return AveragePool(
            channels=data["channels"].item(),
            filter_size=tuple(data["filter_size"]), #type: ignore
            stride=data["stride"].item(),
            padding=data["padding"].item()
        )