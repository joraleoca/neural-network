from dataclasses import dataclass, field

import numpy as np

from .optimizer import Optimizer
from core import Tensor, op, constants as c
from structure import Layer


@dataclass(slots=True)
class Adam(Optimizer):
    """Adam optimizer."""

    b1: float = field(default=0.9)
    b2: float = field(default=0.999)

    mw: list[Tensor[np.floating]] = field(init=False, default_factory=list)
    vw: list[Tensor[np.floating]] = field(init=False, default_factory=list)

    mb: list[Tensor[np.floating]] = field(init=False, default_factory=list)
    vb: list[Tensor[np.floating]] = field(init=False, default_factory=list)

    _tw: int = field(init=False, default=0)
    _tb: int = field(init=False, default=0)

    def __post_init__(self):
        if not (0 <= self.b1 < 1) or not (0 <= self.b2 < 1):
            raise ValueError(
                "The betas must be between 0 (inclusive) and 1 (exclusive)"
            )

    def __call__(self, lr: float, *, layers: list[Layer]) -> None:
        weights = []
        biases = []
        weights_grad = []
        biases_grad = []

        for layer in layers:
            weights.append(layer.weights)
            biases.append(layer.biases)

            weights_grad.append(layer.weights_grad)
            biases_grad.append(layer.biases_grad)

        self._optimize(lr, weights_grad, params=weights, weights=True)
        self._optimize(lr, biases_grad, params=biases, weights=False)

    def _optimize(
        self,
        lr: float,
        gradients: list[Tensor[np.floating]],
        params: list[Tensor[np.floating]],
        weights: bool,
    ) -> None:
        """
        Optimizes the parameters using the Adam optimization algorithm.

        Args:
            lr (float): Learning rate.
            gradients (list[Tensor[np.floating]]): List of gradients for each parameter.
            params (list[Tensor[np.floating]]): List of parameters to be optimized.
            weights (bool): Flag indicating whether to update weights or biases.
        """
        if weights:
            self._tw += 1
            t = self._tw
            m = self.mw
            v = self.vw
        else:
            self._tb += 1
            t = self._tb
            m = self.mb
            v = self.vb

        if not m or not v:
            m[:] = [op.zeros_like(g, dtype=float) for g in gradients]
            v[:] = [op.zeros_like(g, dtype=float) for g in gradients]

        m[:] = [self.b1 * m[i] + (1 - self.b1) * g for i, g in enumerate(gradients)]

        v[:] = [self.b2 * v[i] + (1 - self.b2) * g**2 for i, g in enumerate(gradients)]

        lr = lr * np.sqrt(1 - self.b2**t) / (1 - self.b1**t)

        update = [lr * m_i / (np.sqrt(v_i) + c.EPSILON) for m_i, v_i in zip(m, v)]

        for i, upd in enumerate(update):
            # Done this way to avoid creating new tensors
            params[i].requires_grad = False
            params[i] -= upd
            params[i].requires_grad = True
