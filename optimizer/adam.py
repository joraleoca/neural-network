import numpy as np

from .optimizer import Optimizer
from core import Tensor, op, constants as c


class Adam(Optimizer):
    """Adam optimizer."""

    __slots__ = [
        "b1",
        "b2",
        "mw",
        "vw",
        "mb",
        "vb",
        "_tw",
        "_tb",
    ]

    b1: float
    b2: float

    mw: list[Tensor[np.floating]]
    vw: list[Tensor[np.floating]]

    mb: list[Tensor[np.floating]]
    vb: list[Tensor[np.floating]]

    _tw: int
    _tb: int

    def __init__(self, b1: float = 0.9, b2: float = 0.999):
        if not (0 <= b1 < 1) or not (0 <= b2 < 1):
            raise ValueError(
                "The betas must be between 0 (inclusive) and 1 (exclusive)"
            )

        self.b1 = b1
        self.b2 = b2
        self.mw = []
        self.vw = []
        self.mb = []
        self.vb = []
        self._tw = 0
        self._tb = 0

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

    def optimize_weights(
        self,
        lr: float,
        gradients: list[Tensor[np.floating]],
        *,
        weights: list[Tensor[np.floating]],
    ) -> None:
        return self._optimize(lr, gradients, params=weights, weights=True)

    def optimize_biases(
        self,
        lr: float,
        gradients: list[Tensor[np.floating]],
        *,
        biases: list[Tensor[np.floating]],
    ) -> None:
        return self._optimize(lr, gradients, params=biases, weights=False)
