from dataclasses import dataclass

from .scheduler import Scheduler


@dataclass(slots=True)
class FactorScheduler(Scheduler):
    """
    Learning rate scheduler that reduces the learning rate by a factor.

    Attributes:
        learning_rate (float): Initial learning rate
        factor_lr (float): Factor to multiply learning rate by each update
        min_lr (float): Minimum learning rate
        patience_update (int): Number of epochs to wait before updating.
    """

    learning_rate: float = 0.01
    factor_lr: float = 1.0
    min_lr: float = 1e-7

    patience_update: int = 5

    def __post_init__(self):
        """Validate initialization parameters."""
        if not (0 < self.factor_lr <= 1):
            raise ValueError(
                f"The factor must be between 0 (exclusive) and 1 (inclusive). Got {self.factor_lr}"
            )
        if self.learning_rate <= 0:
            raise ValueError(
                f"The learning rate must be greater than 0. Got {self.learning_rate}"
            )
        if self.min_lr <= 0:
            raise ValueError(
                f"The minimum learning rate must be greater than 0. Got {self.min_lr}"
            )
        if self.patience_update < 0:
            raise ValueError(
                f"The patience to update the learning rate must be non-negative. Got {self.patience_update}"
            )

    def update(self, epoch: int) -> None:
        """
        Updates the learning rate based on the current epoch.
        Args:
            epoch (int): The current epoch number. Must be a non-negative integer.
        Raises:
            ValueError: If the epoch is a negative integer.
        """
        if epoch < 0:
            raise ValueError(f"The epoch must be a non-negative integer. Got {epoch}")

        if self.patience_update == 0 or epoch % self.patience_update == 0:
            self.learning_rate = max(self.min_lr, self.learning_rate * self.factor_lr)
