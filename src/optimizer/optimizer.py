from abc import ABC, abstractmethod

import numpy as np

from src.constants import EPSILON
from src.scheduler import Scheduler
from src.tensor import Tensor


class Optimizer(ABC):
    """
    Abstract base class for neural network optimizers.
    """

    __slots__ = "lr", "_params"

    lr: Scheduler | float
    _params: list[Tensor]

    MAX_DELTA_NORM: float = 1.0

    def __init__(self, params: list[Tensor], lr: Scheduler | float) -> None:
        """
        Initialize the optimizer.

        Args:
            params (list[Tensor]): Parameters to optimize.
            lr (Scheduler | float): Learning rate or learning rate scheduler.
        """
        self.lr = lr
        self._params = params

    @Tensor.no_grad()
    def step(self):
        """
        Optimizes the parameters of the given layers.

        Args:
            lr (float): Learning rate.
        """
        for param in self._params:
            if param.grad is None:
                raise ValueError("Gradients are required for optimization.")

        # Gradient normalization
        global_norm = np.sqrt(sum(np.linalg.norm(d.grad) ** 2 for d in self._params))  # type: ignore
        if global_norm > self.MAX_DELTA_NORM:
            clip_factor = self.MAX_DELTA_NORM / (global_norm + EPSILON)

            for param in self._params:
                param.grad *= clip_factor

        if isinstance(self.lr, Scheduler):
            self._optimize(self.lr.step())
        else:
            self._optimize(self.lr)

    def zero_grad(self) -> None:
        """
        Zeroes the gradients of the parameters.
        """
        for param in self._params:
            param.zero_grad()

    @abstractmethod
    def _optimize(
        self,
        lr: float,
    ) -> None:
        """
        Optimizes the parameters.
        Args:
            lr: learning rate.
        """
        raise NotImplementedError("Optimizer is an abstract class and should not be instantiated directly.")
