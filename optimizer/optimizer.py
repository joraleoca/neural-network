from abc import ABC, abstractmethod

import numpy as np

from core import Tensor


class Optimizer(ABC):
    """
    Abstract base class for neural network optimizers.
    """

    @abstractmethod
    def optimize_weights(
        self,
        lr: float,
        gradients: list[Tensor[np.floating]],
        *,
        weights: list[Tensor[np.floating]],
    ) -> None:
        """
        Optimize the neural network weights in-place based on the provided gradients.
        Args:
            lr (float): Learning rate.
            gradients (list[Tensor[np.floating]]): The gradients of the loss function with respect to the weights.
            weights (list[Tensor[np.floating]]): The current weights of the neural network.
        """
        pass

    @abstractmethod
    def optimize_biases(
        self,
        lr: float,
        gradients: list[Tensor[np.floating]],
        *,
        biases: list[Tensor[np.floating]],
    ) -> None:
        """
        Optimize the neural network biases in-place based on the provided gradients.
        Args:
            lr (float): Learning rate.
            gradients (list[Tensor[np.floating]]): The gradients of the loss function with respect to the biases.
            biases (list[Tensor[np.floating]]): The current biases of the neural network.
        """
        pass
