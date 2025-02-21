from dataclasses import dataclass, field

import numpy as np

from src.core import Tensor
import src.constants as c
from src.core import op
from .optimizer import Optimizer


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

    def _optimize_weights(self, lr: float, weights: list[Tensor], gradients: list[Tensor]) -> None:
        self._tw += 1
        self._optimize(lr, weights, gradients, self._tw, self.mw, self.vw)

    def _optimize_biases(
            self,
            lr: float,
            biases: list[Tensor],
            gradients: list[Tensor],
    ) -> None:
        self._tb += 1
        self._optimize(lr, biases, gradients, self._tb, self.mb, self.vb)

    def _optimize(
            self,
            lr: float,
            params: list[Tensor],
            gradients: list[Tensor],
            t: float,
            m: list[Tensor[np.floating]],
            v: list[Tensor[np.floating]],
    ) -> None:
        """
        Optimize the parameters based on their gradients.

        Args:
            lr: learning rate.
            params: parameters to optimize.
            gradients: gradients to optimize.
            t: t of those params
            m: m of those params
            v: v of those params
        """
        if not m or not v:
            m[:] = [op.zeros_like(g, requires_grad=False) for g in gradients]
            v[:] = [op.zeros_like(g, requires_grad=False) for g in gradients]

        for m_i, v_i, g in zip(m, v, gradients):
            m_i *= self.b1
            m_i += (1 - self.b1) * g
            
            v_i *= self.b2
            v_i += (1 - self.b2) * g**2

        lr *= np.sqrt(1 - self.b2**t) / (1 - self.b1**t)

        update = [lr * m_i / (np.sqrt(v_i) + c.EPSILON) for m_i, v_i in zip(m, v)]

        for i, upd in enumerate(update):
            # Done this way to avoid creating new tensors using in-place operation
            params[i].requires_grad = False
            params[i] -= upd
            params[i].requires_grad = True
