import numpy as np

from src.tensor import Tensor
from src.scheduler import Scheduler

from .optimizer import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer with momentum and Nesterov acceleration support."""

    __slots__ = "momentum", "weight_decay", "nesterov", "_iteration", "_last_gradient"

    momentum: float
    weight_decay: float
    nesterov: bool

    _iteration: int

    _last_gradient: list[Tensor[np.floating]]

    def __init__(
        self,
        parameters: list[Tensor],
        lr: Scheduler | float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        """
        Initialize the optimizer.

        Args:
            parameters: Parameters to optimize.
            momentum: Momentum factor. Default is 0.0.
            weight_decay: Weight decay factor. Default is 0.0.
            nesterov: Whether to use Nesterov momentum. Default is False.
        """
        if not (0 <= self.momentum <= 1):
            raise ValueError(f"Momentum must be between 0 and 1 inclusive. Got {self.momentum}")
        if not (0 <= self.weight_decay <= 0.1):
            raise ValueError(f"Weight decay must be between 0 and 0.1. Got {self.weight_decay}")

        super().__init__(parameters, lr)

        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov

    def _optimize(self, lr: float) -> None:
        if self.momentum != 0:
            self._iteration += 1

            grads = []
            for last_grad, param in zip(self._last_gradient, self._params):
                grad = self.momentum * last_grad + (1 - self.momentum) * param.grad  # type: ignore

                if self.nesterov:
                    grad *= self.momentum
                    grad += param.grad  # type: ignore

                grads.append(grad)  # type: ignore

            self._last_gradient = grads

        for i, param in enumerate(self._params):
            if self.weight_decay > 0:
                param *= 1 - self.weight_decay

            param -= lr * grads[i]
