from abc import ABC, abstractmethod
from typing import ClassVar, Any

import numpy as np

from src.core import Tensor

class Layer(ABC):
    """Abstract class for a layer in a neural network."""

    required_fields: ClassVar[set[str]]

    @abstractmethod
    def forward(
        self,
        data: Tensor[np.floating],
    ) -> Tensor[np.floating]:
        """
        Forward pass of the layer.
        Args:
            data (Tensor): The input data to the layer.
        Returns:
            Tensor: The output of the layer.
        """
        pass

    @abstractmethod
    def data_to_store(self) -> dict[str, Any]:
        """
        Data to store of the layer.
        Returns:
            dict[str, Any]: Data to store of the layer.
        """
        pass

    @staticmethod
    @abstractmethod
    def from_data(data: dict[str, Any]) -> "Layer":
        """
        Create a new layer from the given data.

        Args:
            data (dict[str, Any]): Data to create the layer from.
        Returns:
            Layer: The created layer.
        """
        pass
