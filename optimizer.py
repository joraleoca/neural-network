import numpy as np
from numpy.typing import NDArray

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Final
from copy import deepcopy


class Optimizer(ABC):
    learning_rate: float

    EPS: Final[float] = 1e-8

    @abstractmethod
    def optimize(
        self,
        gradients: list[NDArray[np.floating[Any]]],
        *,
        weights: list[NDArray[np.floating[Any]]],
        biases: list[NDArray[np.floating[Any]]],
        output_layers: list[NDArray[np.floating[Any]]] | None,
    ) -> None:
        """
        Optimize the neural network parameters based on the provided gradients.
        Args:
            gradients (list[NDArray[np.floating[Any]]]): The gradients of the loss function with respect to the parameters.
            weights (list[NDArray[np.floating[Any]]]): The current weights of the neural network.
            biases (list[NDArray[np.floating[Any]]]): The current biases of the neural network.
        Returns:
            None: weights and biases are changed in place.
        """
        pass


@dataclass
class SGD(Optimizer):
    learning_rate: float = field(default=0.001, kw_only=True)

    momentum: float = field(default=0, kw_only=True)
    weight_decay: float = field(default=0, kw_only=True)

    nesterov: bool = field(default=False, kw_only=True)

    _t: int = field(init=False, default=0)
    _last_gradient: list[NDArray[np.floating[Any]]] = field(
        init=False, default_factory=list
    )

    def optimize(
        self,
        gradients: list[NDArray[np.floating[Any]]],
        *,
        weights: list[NDArray[np.floating[Any]]],
        biases: list[NDArray[np.floating[Any]]],
        output_layers: list[NDArray[np.floating[Any]]],
    ) -> None:
        if self.momentum != 0:
            self._t += 1

        if self._t > 1:
            self._last_gradient = [
                self.momentum * lg + g for lg, g in zip(self._last_gradient, gradients)
            ]
        else:
            self._last_gradient = deepcopy(gradients)

        if self.nesterov:
            gradients = [
                self.momentum * lg + g for lg, g in zip(self._last_gradient, gradients)
            ]
        else:
            gradients = deepcopy(self._last_gradient)

        for i in reversed(range(len(weights))):
            weight_update = self.learning_rate * (gradients[i] @ output_layers[i].T)
            bias_update = self.learning_rate * gradients[i]

            weights[i] -= weight_update
            biases[i] -= bias_update


@dataclass
class Adam(Optimizer):
    learning_rate: float = field(default=0.001, kw_only=True)

    b1: float = field(default=0.9, kw_only=True)
    b2: float = field(default=0.999, kw_only=True)

    m: list[NDArray[np.floating[Any]]] = field(init=False, default_factory=list)
    v: list[NDArray[np.floating[Any]]] = field(init=False, default_factory=list)

    _t: int = field(init=False, default=0)

    b1: float = field(default=0.9, kw_only=True)
    b2: float = field(default=0.999, kw_only=True)

    def optimize(
        self,
        gradients: list[NDArray[np.floating[Any]]],
        *,
        weights: list[NDArray[np.floating[Any]]],
        biases: list[NDArray[np.floating[Any]]],
        output_layers: None = None,
    ) -> None:
        self._t += 1

        if not self.m or not self.v:
            self.m = [np.zeros_like(g, dtype=float) for g in gradients]
            self.v = [np.zeros_like(g, dtype=float) for g in gradients]

        self.m = [
            self.b1 * self.m[i] + (1 - self.b1) * g for i, g in enumerate(gradients)
        ]

        self.v = [
            self.b2 * self.v[i] + (1 - self.b2) * g**2 for i, g in enumerate(gradients)
        ]

        lr = self.learning_rate * np.sqrt(1 - self.b2**self._t) / (1 - self.b1**self._t)

        update = [lr * m / (np.sqrt(v) + self.EPS) for m, v in zip(self.m, self.v)]

        for i in range(len(weights)):
            weights[i] -= update[i]
            biases[i] -= np.mean(update[i], axis=1, keepdims=True)
