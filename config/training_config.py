from dataclasses import dataclass, field

from scheduler import Scheduler, FactorScheduler


@dataclass(slots=True)
class TrainingConfig:
    """
    TrainingConfig class for configuring the training parameters of a neural network.
    Attributes:
        lr (Scheduler): Learning rate scheduler.
        epochs (int): Number of training epochs.
        patience_stop (int): Patience for early stopping, 0 is no stop.
        min_delta (float): Minimum change in loss to be considered an improvement.
        debug (bool): Flag to enable debugging mode.
    """

    lr: Scheduler = field(default_factory=FactorScheduler)

    epochs: int = 10_000
    patience_stop: int = 10  # Early stopping patience
    min_delta: float = 1e-4  # Minimum change in loss to be considered an improvement
    debug: bool = False
    store: bool = False

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
