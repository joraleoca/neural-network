from dataclasses import dataclass, field

import numpy as np

from .scheduler import Scheduler


@dataclass(slots=True)
class CosineScheduler(Scheduler):
    """
    Learning rate scheduler that reduces the learning rate by a factor.

    Attributes:
        learning_rate (float): Initial learning rate
        factor_lr (float): Factor to multiply learning rate by each update
        min_lr (float): Minimum learning rate
        max_steps (int): Max steps until learning rate is min_lr
    """

    max_steps: int

    learning_rate: float = 0.01
    min_lr: float = 1e-7

    _initial_lr: float = field(init=False)
    _t: int = field(init=False, default=0)

    def __post_init__(self):
        """Validate initialization parameters."""
        if self.learning_rate <= 0:
            raise ValueError(
                f"The learning rate must be greater than 0. Got {self.learning_rate}"
            )
        if self.min_lr <= 0:
            raise ValueError(
                f"The minimum learning rate must be greater than 0. Got {self.min_lr}"
            )
        self._initial_lr = self.learning_rate

    def update(self) -> None:
        self.learning_rate = self.min_lr + ((self._initial_lr - self.min_lr) / 2) * (
            1 + np.cos((np.pi * self._t / self.max_steps))
        )
        self._t += 1
