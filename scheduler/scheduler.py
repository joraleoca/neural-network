from abc import ABC, abstractmethod


class Scheduler(ABC):
    """
    Scheduler is an abstract base class for learning rate schedulers.
    Attributes:
        learning_rate (float): The current learning rate.
        patience_update (int): The number of epochs to wait before updating the learning rate.
                               Default is 0, meaning the learning rate is always updated.
    """

    learning_rate: float
    patience_update: int = 0  # Used by FactorScheduler. In other schedulers, the default is 0, meaning is always updated.

    @abstractmethod
    def update(self) -> None:
        """
        Update the learning rate in-place.
        """
        pass
