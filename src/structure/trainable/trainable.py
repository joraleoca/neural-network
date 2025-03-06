from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.core import Tensor
from src.initialization import Initializer

from ..layer import Layer


class Trainable(Layer, ABC):
    __slots__ = [
        "weights",
        "biases",
        "rng",
        "_requires_grad",
        "_initializer",
    ]

    weights: Tensor[np.floating]
    biases: Tensor[np.floating]

    rng: Any

    _requires_grad: bool

    # Store only to use the first time forward is called, after that deleted
    _initializer: Initializer | None

    def __init__(self, initializer: Initializer | None = None, *, requires_grad: bool = False, rng: Any = None) -> None:
        self._initializer = initializer
        self._requires_grad = requires_grad
        self.rng = rng

        # Create the weights and biases until they are initialized so they can be referenced after the first forward pass
        self.weights = Tensor([])
        self.biases = Tensor([])

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool) -> None:
        self._requires_grad = requires_grad
        self.weights.requires_grad = requires_grad
        self.biases.requires_grad = requires_grad

    @property
    def initializer(self) -> Initializer | None:
        return self._initializer

    @initializer.setter
    def initializer(self, initializer: Initializer) -> None:
        self._initializer = initializer

    def zero_grad(self) -> None:
        """Clears the gradients of the weights and biases."""
        self.weights.zero_grad()
        self.biases.zero_grad()

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
