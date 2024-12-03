from abc import ABC, abstractmethod

from structure import Layer


class Optimizer(ABC):
    """
    Abstract base class for neural network optimizers.
    """

    @abstractmethod
    def __call__(self, lr: float, *, layers: list[Layer]):
        """
        Optimizes the parameters of the given layers.

        Args:
            lr (float): Learning rate.
            layers (list[Layer]): Layers to optimize.
        """
        pass
