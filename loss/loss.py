from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from encode import Encoder


class Loss(ABC):
    """
    Abstract base class for loss functions in a neural network.
    """

    def __call__(
        self,
        expected: NDArray[np.floating[Any]],
        predicted: NDArray[np.floating[Any]],
    ) -> np.floating[Any]:
        """
        Calculates the loss between the expected and predicted values.

        Parameters:
            expected (NDArray[np.floating[Any]]): The expected values.
            predicted (NDArray[np.floating[Any]]): The predicted values.

        Returns:
            np.floating[Any]: The loss value.
        """
        return self.loss(expected, predicted)

    @abstractmethod
    def loss(
        self,
        expected: NDArray[np.floating[Any]],
        predicted: NDArray[np.floating[Any]],
    ) -> np.floating[Any]:
        """
        Calculates the loss between the expected and predicted values.

        Parameters:
            expected (NDArray[np.floating[Any]]): The expected values.
            predicted (NDArray[np.floating[Any]]): The predicted values.

        Returns:
            np.floating[Any]: The loss value.
        """
        pass

    @abstractmethod
    def gradient(
        self,
        expected: NDArray[np.floating[Any]],
        predicted: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """
        Compute the gradient of the loss function with respect to the predicted values.

        Args:
            expected (NDArray[np.floating[Any]]): The expected values.
            predicted (NDArray[np.floating[Any]]): The predicted values.

        Returns:
            np.floating[Any]: The gradient of the loss function.
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
