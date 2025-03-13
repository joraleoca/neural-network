from abc import ABC, abstractmethod

import numpy as np

from src.tensor import Tensor


class Layer(ABC):
    """Abstract class for a layer in a neural network."""

    @abstractmethod
    def __call__(
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
