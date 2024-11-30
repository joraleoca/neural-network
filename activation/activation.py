from abc import ABC, abstractmethod

from core import Tensor


class ActivationFunction(ABC):
    """
    Abstract base class defining the interface for neural network activation functions.

    All activation functions must implement the __call__() method.
    """

    @abstractmethod
    def __call__(
        self,
        *args: Tensor,
    ) -> Tensor:
        pass
