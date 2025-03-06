from dataclasses import dataclass
from pathlib import Path

from src.optimizer import Optimizer
from src.loss import Loss


@dataclass(slots=True)
class TrainingConfig:
    """
    TrainingConfig class for configuring the training parameters of a neural network.
    Attributes:
        loss (Loss): Loss function.
        epochs (int): Number of training epochs.
        patience_stop (int): Patience for early stopping, 0 is no stop.
        min_delta (float): Minimum change in loss to be considered an improvement.
        optimizer (Optimizer): Optimizer for the model.
        batch_size (int): Batch size for training.
        debug (bool): Flag to enable debugging mode.
        store (str | Path | None): Path to store the training results or None if not store.
    """

    loss: Loss
    optimizer: Optimizer

    epochs: int = 10_000
    patience_stop: int = 10  # Early stopping patience
    min_delta: float = 1e-4  # Minimum change in loss to be considered an improvement

    batch_size: int = 1

    debug: bool = False
    store: str | Path | None = None

    def __post_init__(self):
        """Validate initialization parameters."""
        if self.epochs <= 0:
            raise ValueError(f"Epochs must be greater than 0. Got {self.epochs}.")
        if self.patience_stop < 0:
            raise ValueError(f"The patience to stop must be non negative. Got {self.patience_stop}.")
        if self.min_delta <= 0:
            raise ValueError(f"Min delta must be greater than 0. Got {self.min_delta}")
        if self.batch_size < 1:
            raise ValueError(f"The batch size must be positive. Got {self.batch_size}")
