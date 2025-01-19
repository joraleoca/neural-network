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
        "_weights_grad",
        "_biases_grad",
        "_times_grad",
        "_initializer",
    ]

    weights: Tensor[np.floating]
    biases: Tensor[np.floating]

    activation_function: ActivationFunction | None

    rng: Any

    _requires_grad: bool

    _weights_grad: Tensor[np.floating] | None
    _biases_grad: Tensor[np.floating] | None

    _times_grad: int

    # Store only to use the first time forward is called, after that deleted
    _initializer: Initializer | None

    def __init__(self, requires_grad: bool = False) -> None:
        self._requires_grad = requires_grad
        self._weights_grad = None
        self._biases_grad = None
        self._times_grad = 0

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
            # Clean memory
            self._weights_grad = None
            self._biases_grad = None

    @property
    def weights_grad(self) -> Tensor[np.floating]:
        """Returns the accumulated gradients of the weights."""
        return self._weights_grad / self._times_grad

    @property
    def biases_grad(self) -> Tensor[np.floating]:
        """Returns the accumulated gradients of the biases."""
        return self._biases_grad / self._times_grad

    @property
    def initializer(self) -> Initializer | None:
        """Returns the initializer of the layer."""
        return self._initializer

    @initializer.setter
    def initializer(self, initializer: Initializer) -> None:
        """Sets the initializer of the layer."""
        self._initializer = initializer

    def backward(self) -> None:
        """
        Accumulates the gradients of the layer parameters.\n
        The gradients of these are cleared.
        """
        if self._weights_grad is None or self._biases_grad is None:
            self._weights_grad = deepcopy(self.weights.grad)
            self._biases_grad = deepcopy(self.biases.grad)
            self._times_grad = 1
        else:
            self._weights_grad += self.weights.grad
            self._biases_grad += self.biases.grad
            self._times_grad += 1

        self.weights.clear_grad()
        self.biases.clear_grad()

    def clear_params_grad(self) -> None:
        """Clears the gradient of the layer or initializes it if it does not exist."""
        self.weights.clear_grad()
        self.biases.clear_grad()
        self._weights_grad.fill(0.0)
        self._biases_grad.fill(0.0)
        self._times_grad = 0

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
