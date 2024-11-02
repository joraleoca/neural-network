from typing import Any
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Optimizer(ABC):
    """
    Abstract base class for neural network optimizers.
    """

    @abstractmethod
    def optimize_weights(
        self,
        lr: float,
        gradients: list[NDArray[np.floating[Any]]],
        *,
        weights: list[NDArray[np.floating[Any]]],
    ) -> None:
        """
        Optimize the neural network weights in-place based on the provided gradients.
        Args:
            lr (float): Learning rate.
            gradients (list[NDArray[np.floating[Any]]]): The gradients of the loss function with respect to the weights.
            weights (list[NDArray[np.floating[Any]]]): The current weights of the neural network.
        """
        pass

    @abstractmethod
    def optimize_biases(
        self,
        lr: float,
        gradients: list[NDArray[np.floating[Any]]],
        *,
        biases: list[NDArray[np.floating[Any]]],
    ) -> None:
        """
        Optimize the neural network biases in-place based on the provided gradients.
        Args:
            lr (float): Learning rate.
            gradients (list[NDArray[np.floating[Any]]]): The gradients of the loss function with respect to the biases.
            biases (list[NDArray[np.floating[Any]]]): The current biases of the neural network.
        """
        pass
