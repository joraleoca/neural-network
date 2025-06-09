from .scheduler import Scheduler


class FactorScheduler(Scheduler):
    """
    Learning rate scheduler that reduces the learning rate by a factor.
    """

    __slots__ = "factor_lr", "min_lr", "patience_update"

    def __init__(self, lr: float, factor_lr: float = 0.01, min_lr: float = 1e-7, patience_update: int = 5) -> None:
        if not (0 < factor_lr <= 1):
            raise ValueError(f"The factor must be between 0 (exclusive) and 1 (inclusive). Got {factor_lr}")
        if lr <= 0:
            raise ValueError(f"The learning rate must be greater than 0. Got {lr}")
        if min_lr <= 0:
            raise ValueError(f"The minimum learning rate must be greater than 0. Got {min_lr}")
        if patience_update < 0:
            raise ValueError(f"The patience to update the learning rate must be non-negative. Got {patience_update}")

        super().__init__(lr)

        self.factor_lr = factor_lr
        self.min_lr = min_lr
        self.patience_update = patience_update

    def step(self) -> float:
        self.iterations += 1

        if self.learning_rate > self.min_lr and (
            self.patience_update == 0 or self.iterations % self.patience_update == 0
        ):
            self.learning_rate = max(self.min_lr, self.learning_rate * self.factor_lr)

        return self.learning_rate
