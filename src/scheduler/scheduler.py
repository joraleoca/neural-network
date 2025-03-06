from abc import ABC, abstractmethod


class Scheduler(ABC):
    """
    Scheduler is an abstract base class for learning rate schedulers.
    """

    __slots__ = "learning_rate", "iterations"

    learning_rate: float
    iterations: int

    def __init__(self, lr: float) -> None:
        """
        Initialize the scheduler.

        Args:
            lr (float): The initial learning rate.
        """
        self.learning_rate = lr
        self.iterations = 0

    @abstractmethod
    def step(self) -> float:
        """
        Update the learning rate based on the current epoch.

        Returns:
            float: The updated learning
        """
        raise NotImplementedError(
            f"{type(self).__name__} is an abstract class and should not be instantiated directly."
        )
