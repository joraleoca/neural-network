import numpy as np

from .scheduler import Scheduler


class CosineScheduler(Scheduler):
    """
    CosineScheduler is a learning rate scheduler that adjusts the learning rate
    following a cosine schedule.
    """

    __slots__ = "max_steps", "min_lr", "cyclic", "_initial_lr"

    def __init__(self, learning_rate: float, min_lr: float = 1e-7, max_steps: int = 100, cyclic: bool = False) -> None:
        """
        Initialize the cosine learning rate scheduler.

        Args:
            learning_rate (float): The initial learning rate.
            min_lr (float): The minimum learning rate.
            max_steps (int): The number of steps to reach the minimum learning rate.
            cyclic (bool): Whether the learning rate should be cyclic.
        """
        if learning_rate <= 0:
            raise ValueError(f"The learning rate must be greater than 0. Got {learning_rate}")
        if min_lr <= 0:
            raise ValueError(f"The minimum learning rate must be greater than 0. Got {min_lr}")
        if min_lr > learning_rate:
            raise ValueError(
                f"The minimum learning rate must be less than the initial learning rate. Got {min_lr} > {learning_rate}"
            )
        super().__init__(learning_rate)
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.cyclic = cyclic
        self._initial_lr = learning_rate

    def step(self) -> float:
        if self.cyclic or self.iterations < self.max_steps:
            self.learning_rate = self.min_lr + ((self._initial_lr - self.min_lr) / 2) * (
                1 + np.cos((np.pi * self.iterations / self.max_steps))
            )

        self.iterations += 1
        return self.learning_rate
