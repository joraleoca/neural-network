from abc import ABC, abstractmethod

import numpy as np

from src.structure import Trainable
from src.core import Tensor
from src.constants import EPSILON


class Optimizer(ABC):
    """
    Abstract base class for neural network optimizers.
    """

    MAX_DELTA_NORM: float = 5.0

    def __call__(self, lr: float, *, layers: list[Trainable]):
        """
        Optimizes the parameters of the given layers.

        Args:
            lr (float): Learning rate.
            layers (list[Trainable]): Trainable layers to optimize.
        """

        weights, biases = list(), list()
        weights_grad, biases_grad = list(), list()

        for layer in layers:
            weights.append(layer.weights)
            biases.append(layer.biases)

            weights_grad.append(layer.weights_grad)
            biases_grad.append(layer.biases_grad)

        # Gradient normalization
        global_norm = np.sqrt(
            sum(np.linalg.norm(d) ** 2 for d in weights_grad) +
            sum(np.linalg.norm(d) ** 2 for d in biases_grad)
        )
        if global_norm > self.MAX_DELTA_NORM:
            clip_factor = self.MAX_DELTA_NORM / (global_norm + EPSILON)
            
            for i in range(len(weights_grad)):
                weights_grad[i] *= clip_factor
                biases_grad[i] *= clip_factor

        self._optimize_weights(lr, weights, weights_grad)
        self._optimize_biases(lr, biases, biases_grad)

    @abstractmethod
    def _optimize_weights(
        self,
        lr: float,
        weights: list[Tensor],
        gradients: list[Tensor],
    ) -> None:
        """
        Optimizes the weights of the given layers.
        Args:
            lr: learning rate.
            weights (list[Tensor[np.floating]]): Weights.
            gradients (list[Tensor[np.floating]]): Gradients.
        """
        pass

    @abstractmethod
    def _optimize_biases(
        self,
        lr: float,
        biases: list[Tensor],
        gradients: list[Tensor],
    ) -> None:
        """
        Optimizes the biases of the given layers.
        Args:
            lr (float): learning rate.
            biases (list[Tensor[np.floating]]): Biases.
            gradients (list[Tensor[np.floating]]): Gradients.
        """
        pass
