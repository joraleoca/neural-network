import numpy as np

import src.constants as c
from src.core import Tensor, op
from src.scheduler import Scheduler

from .optimizer import Optimizer


class Adam(Optimizer):
    """Adam optimizer."""

    __slots__ = "b1", "b2", "_m", "_v", "_t"

    b1: float
    b2: float

    _m: list[Tensor[np.floating]]
    _v: list[Tensor[np.floating]]

    _t: int

    def __init__(self, parameters: list[Tensor], lr: Scheduler | float, b1: float = 0.9, b2: float = 0.999) -> None:
        if not (0 <= b1 < 1) or not (0 <= b2 < 1):
            raise ValueError(f"The betas must be between 0 (inclusive) and 1 (exclusive). Got b1: {b1} and b2: {b2}")

        super().__init__(parameters, lr)

        self.b1, self.b2 = b1, b2
        self._t = 0

        self._m = []
        self._v = []

    def _optimize(self, lr: float) -> None:
        if not self._m or not self._v:
            self._m = [op.zeros_like(param) for param in self._params]
            self._v = [op.zeros_like(param) for param in self._params]

        self._t += 1

        for m, v, param in zip(self._m, self._v, self._params):
            m *= self.b1
            v *= self.b2

            m += (1 - self.b1) * param.grad  # type: ignore
            v += (1 - self.b2) * param.grad**2  # type: ignore

        lr *= np.sqrt(1 - self.b2**self._t) / (1 - self.b1**self._t)

        for param, m, v in zip(self._params, self._m, self._v):
            param.requires_grad = False
            param -= lr * m / (np.sqrt(v) + c.EPSILON)
            param.requires_grad = True
