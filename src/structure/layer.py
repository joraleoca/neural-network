from abc import ABC, abstractmethod

import numpy as np

from src.tensor import Tensor


class Layer(ABC):
    """Abstract class for a layer in a neural network."""

    @abstractmethod
    def __call__(
        self,
        *args: Tensor[np.floating],
    ) -> Tensor[np.floating]:
        """
        Forward pass of the layer.
        Args:
            args (*Tensor): The input data to the layer.
        Returns:
            Tensor: The output of the layer.
        """
        pass
