from dataclasses import dataclass, field
from pathlib import Path

from scheduler import Scheduler, FactorScheduler
from optimizer import Optimizer, SGD
from loss import Loss


@dataclass(slots=True)
class TrainingConfig:
    """
    TrainingConfig class for configuring the training parameters of a neural network.
    Attributes:
        loss (Loss): Loss function.
        lr (Scheduler): Learning rate scheduler.
        epochs (int): Number of training epochs.
        patience_stop (int): Patience for early stopping, 0 is no stop.
        min_delta (float): Minimum change in loss to be considered an improvement.
        optimizer (Optimizer): Optimizer for the model.
        batch_size (int): Batch size for training.
        dropout (float): Dropout rate
        debug (bool): Flag to enable debugging mode.
        store (str | Path | None): Path to store the training results or None if not store.
    """

    loss: Loss

    lr: Scheduler = field(default_factory=FactorScheduler)

    epochs: int = 10_000
    patience_stop: int = 10  # Early stopping patience
    min_delta: float = 1e-4  # Minimum change in loss to be considered an improvement

    optimizer: Optimizer = field(default_factory=SGD)
    batch_size: int = 1
    dropout: float = 0.0

    debug: bool = False
    store: str | Path | None = None

    def __post_init__(self):
        """Validate initialization parameters."""
        if not isinstance(self.lr, Scheduler):
            raise TypeError(
                "The learning rate must be an instance of Scheduler. Default scheduler is FactorScheduler with a factor of 1."
            )

        if self.epochs <= 0:
            raise ValueError("Epochs must be greater than 0")

        if self.patience_stop < 0:
            raise ValueError(
                f"The patience to stop must be non negative. Got {self.patience_stop}."
            )

        if self.min_delta <= 0:
            raise ValueError("Min delta must be greater than 0")

        if self.batch_size < 1:
            raise ValueError(
                f"The batch size must be positive number. Got {self.batch_size}"
            )

        if not 0 <= self.dropout < 1:
            raise ValueError("The dropout rate must be between 0 and 1.")
