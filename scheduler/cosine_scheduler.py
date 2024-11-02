from dataclasses import dataclass, field

import numpy as np

from .scheduler import Scheduler


@dataclass(slots=True)
class CosineScheduler(Scheduler):
    """
    CosineScheduler is a learning rate scheduler that adjusts the learning rate
    following a cosine schedule.

    Attributes:
        max_steps (int): The total number of steps (epochs) for the scheduler.
        learning_rate (float): The initial learning rate. Default is 0.01.
        min_lr (float): The minimum learning rate. Default is 1e-7.
    """

    max_steps: int

    learning_rate: float = 0.01
    min_lr: float = 1e-7

    _initial_lr: float = field(init=False)

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
        if self.min_lr > self.learning_rate:
            raise ValueError(
                f"The minimum learning rate must be less than the initial learning rate. Got {self.min_lr} > {self.learning_rate}"
            )

        self._initial_lr = self.learning_rate

    def update(self, epoch: int) -> None:
        self.learning_rate = self.min_lr + ((self._initial_lr - self.min_lr) / 2) * (
            1 + np.cos((np.pi * epoch / self.max_steps))
        )
