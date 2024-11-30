from abc import ABC, abstractmethod

import numpy as np

from encode import Encoder
from core import Tensor


class Loss(ABC):
    """
    Abstract base class for loss functions in a neural network.
    """

    @abstractmethod
    def __call__(
        self,
        expected: Tensor[np.floating],
        predicted: Tensor[np.floating],
    ) -> Tensor[np.floating]:
        """
        Calculates the loss between the expected and predicted values.

        Parameters:
            expected (Tensor[np.floating]): The expected values.
            predicted (Tensor[np.floating]): The predicted values.

        Returns:
            Tensor[np.floating]: The loss value.
        """
        pass

    @staticmethod
    @abstractmethod
    def encoder() -> type[Encoder]:
        """
        Defines the encoder function.

        Returns:
            type[Encoder]: The type of the Encoder class.
        """
        pass
