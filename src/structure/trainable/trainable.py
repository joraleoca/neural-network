from abc import ABC, abstractmethod
from typing import Any
from copy import deepcopy

import numpy as np

from ..layer import Layer
from src.core import Tensor
from src.activation import ActivationFunction
from src.initialization import Initializer


class Trainable(Layer, ABC):
    __slots__ = [
        "weights",
        "biases",
        "activation_function",
        "rng",
        "_requires_grad",
        "_initializer",
    ]

    weights: Tensor[np.floating]
    biases: Tensor[np.floating]

    activation_function: ActivationFunction | None

    rng: Any

    _requires_grad: bool

    # Store only to use the first time forward is called, after that deleted
    _initializer: Initializer | None

    def __init__(self, requires_grad: bool = False) -> None:
        self._requires_grad = requires_grad

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool) -> None:
        self._requires_grad = requires_grad
        if hasattr(self, "weights"):
            self.weights.requires_grad = requires_grad
        if hasattr(self, "biases"):
            self.biases.requires_grad = requires_grad

        if not requires_grad:
            self.clear_params_grad()

    @property
    def weights_grad(self) -> Tensor[np.floating]:
        """Returns the accumulated gradients of the weights."""
        return self.weights.grad

    @property
    def biases_grad(self) -> Tensor[np.floating]:
        """Returns the accumulated gradients of the biases."""
        return self.biases.grad

    @property
    def initializer(self) -> Initializer | None:
        """Returns the initializer of the layer."""
        return self._initializer

    @initializer.setter
    def initializer(self, initializer: Initializer) -> None:
        """Sets the initializer of the layer."""
        self._initializer = initializer

    def clear_params_grad(self) -> None:
        """Clears the gradient of the layer or initializes it if it does not exist."""
        self.weights.clear_grad()
        self.biases.clear_grad()

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Returns the number of input features of the layer."""
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Returns the number of output features of the layer."""
        pass
