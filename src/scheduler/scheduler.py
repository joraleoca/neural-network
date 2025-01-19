from abc import ABC, abstractmethod


class Scheduler(ABC):
    """
    Scheduler is an abstract base class for learning rate schedulers.
    """

    learning_rate: float

    @abstractmethod
    def update(self, epoch: int) -> None:
        """
        Update the learning rate based on the current epoch.

        Args:
            epoch (int): The current epoch number.
        """
        pass
