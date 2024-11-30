from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from .optimizer import Optimizer
from core import Tensor


@dataclass(slots=True)
class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer with momentum and Nesterov acceleration support."""

    momentum: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False

    _iteration_weights: int = field(default=0, init=False)
    _iteration_biases: int = field(default=0, init=False)
    _last_weights_gradient: list[Tensor[np.floating]] = field(
        default_factory=list, init=False
    )
    _last_biases_gradient: list[Tensor[np.floating]] = field(
        default_factory=list, init=False
    )

    def __post_init__(self):
        """Validate initialization parameters."""
        if not (0 <= self.momentum <= 1):
            raise ValueError(
                f"Momentum must be between 0 and 1 inclusive. Got {self.momentum}"
            )
        if not (0 <= self.weight_decay <= 0.1):
            raise ValueError(
                f"Weight decay must be between 0 and 0.1. Got {self.weight_decay}"
            )

    def _apply_momentum(
        self,
        gradients: list[Tensor[np.floating]],
        last_gradients: list[Tensor[np.floating]],
        iteration: int,
    ) -> list[Tensor[np.floating]]:
        """
        Apply momentum to gradients.

        Args:
            gradients: Current gradients
            last_gradients: Previous gradients
            iteration: Current iteration number

        Returns:
            Updated gradients with momentum applied
        """
        if self.momentum == 0:
            return deepcopy(gradients)

        if iteration > 1:
            momentum_gradients = [
                self.momentum * lg + g for lg, g in zip(last_gradients, gradients)
            ]
        else:
            momentum_gradients = deepcopy(gradients)

        if self.nesterov:
            return [
                g + self.momentum * lg for lg, g in zip(momentum_gradients, gradients)
            ]
        return momentum_gradients

    def _apply_update(
        self,
        lr: float,
        params: list[Tensor[np.floating]],
        gradients: list[Tensor[np.floating]],
    ) -> None:
        """
        Apply gradient updates to parameters.

        Args:
            params: Parameters to update (weights or biases)
            gradients: Gradients to apply
        """
        for i in reversed(range(len(params))):
            params[i].requires_grad = False
            if self.weight_decay > 0:
                params[i] *= 1 - self.weight_decay
            params[i] -= lr * gradients[i]
            params[i].requires_grad = True

    def optimize_weights(
        self,
        lr: float,
        gradients: list[Tensor[np.floating]],
        *,
        weights: list[Tensor[np.floating]],
    ) -> None:
        if self.momentum != 0:
            self._iteration_weights += 1

        self._last_weights_gradient = self._apply_momentum(
            gradients, self._last_weights_gradient, self._iteration_weights
        )
        self._apply_update(lr, weights, self._last_weights_gradient)

    def optimize_biases(
        self,
        lr: float,
        gradients: list[Tensor[np.floating]],
        *,
        biases: list[Tensor[np.floating]],
    ) -> None:
        if self.momentum != 0:
            self._iteration_biases += 1

        self._last_biases_gradient = self._apply_momentum(
            gradients, self._last_biases_gradient, self._iteration_biases
        )
        self._apply_update(lr, biases, self._last_biases_gradient)
