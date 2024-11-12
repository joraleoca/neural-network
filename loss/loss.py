from abc import ABC, abstractmethod


import numpy as np
from numpy.typing import NDArray

from encode import Encoder


class Loss(ABC):
    """
    Abstract base class for loss functions in a neural network.
    """

    def __call__(
        self,
        expected: NDArray[np.floating],
        predicted: NDArray[np.floating],
    ) -> np.floating:
        """
        Calculates the loss between the expected and predicted values.

        Parameters:
            expected (NDArray[np.floating]): The expected values.
            predicted (NDArray[np.floating]): The predicted values.

        Returns:
            np.floating: The loss value.
        """
        return self.loss(expected, predicted)

    @abstractmethod
    def loss(
        self,
        expected: NDArray[np.floating],
        predicted: NDArray[np.floating],
    ) -> np.floating:
        """
        Calculates the loss between the expected and predicted values.

        Parameters:
            expected (NDArray[np.floating]): The expected values.
            predicted (NDArray[np.floating]): The predicted values.

        Returns:
            np.floating: The loss value.
        """
        pass

    @abstractmethod
    def gradient(
        self,
        expected: NDArray[np.floating],
        predicted: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Compute the gradient of the loss function with respect to the predicted values.

        Args:
            expected (NDArray[np.floating]): The expected values.
            predicted (NDArray[np.floating]): The predicted values.

        Returns:
            np.floating: The gradient of the loss function.
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
