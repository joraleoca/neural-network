from dataclasses import dataclass, field


import numpy as np
from numpy.typing import NDArray

from .optimizer import Optimizer
from core.constants import EPSILON


@dataclass(slots=True)
class Adam(Optimizer):
    b1: float = field(default=0.9, kw_only=True)
    b2: float = field(default=0.999, kw_only=True)

    mw: list[NDArray[np.floating]] = field(init=False, default_factory=list)
    vw: list[NDArray[np.floating]] = field(init=False, default_factory=list)

    mb: list[NDArray[np.floating]] = field(init=False, default_factory=list)
    vb: list[NDArray[np.floating]] = field(init=False, default_factory=list)

    _tw: int = field(init=False, default=0)
    _tb: int = field(init=False, default=0)

    def __post_init__(self):
        """Validate initialization parameters."""
        if not (0 <= self.b1 < 1) or not (0 <= self.b2 < 1):
            raise ValueError(
                "The betas must be between 0 (inclusive) and 1 (exclusive)"
            )

    def _optimize(
        self,
        lr: float,
        gradients: list[NDArray[np.floating]],
        params: list[NDArray[np.floating]],
        weights: bool,
    ) -> None:
        """
        Optimizes the parameters using the Adam optimization algorithm.
        Args:
            lr (float): Learning rate.
            gradients (list[NDArray[np.floating]]): List of gradients for each parameter.
            params (list[NDArray[np.floating]]): List of parameters to be optimized.
            weights (bool): Flag indicating whether to update weights or biases.
        Returns:
            None
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
            m[:] = [np.zeros_like(g, dtype=float) for g in gradients]
            v[:] = [np.zeros_like(g, dtype=float) for g in gradients]

        m[:] = [self.b1 * m[i] + (1 - self.b1) * g for i, g in enumerate(gradients)]

        v[:] = [self.b2 * v[i] + (1 - self.b2) * g**2 for i, g in enumerate(gradients)]

        lr = lr * np.sqrt(1 - self.b2**t) / (1 - self.b1**t)

        update = [lr * m_i / (np.sqrt(v_i) + EPSILON) for m_i, v_i in zip(m, v)]

        for i, upd in enumerate(update):
            params[i] -= upd

    def optimize_weights(
        self,
        lr: float,
        gradients: list[NDArray[np.floating]],
        *,
        weights: list[NDArray[np.floating]],
    ) -> None:
        return self._optimize(lr, gradients, params=weights, weights=True)

    def optimize_biases(
        self,
        lr: float,
        gradients: list[NDArray[np.floating]],
        *,
        biases: list[NDArray[np.floating]],
    ) -> None:
        return self._optimize(lr, gradients, params=biases, weights=False)
