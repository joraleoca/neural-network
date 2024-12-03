from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator

from core import Tensor


class Initializator(ABC):
    """
    Initializator is an abstract base class for initializing neural network weights.
    """

    __slots__ = ["gain"]

    gain: float

    def __init__(self, gain: float = 1):
        self.gain = gain

    @abstractmethod
    def initialize(
        self, shape: tuple[int, ...], *, rng: Generator | None = None
    ) -> Tensor:
        """
        Initializes the neural network with the given structure.

        Args:
            shape (tuple[int, ...]): The shape of the weight matrix.
            rng (Generator | None, optional): A random number generator instance. If None,
                                          the default random generator will be used.

        Returns:
            Tensor: A list containing the weight matrices for each layer.
        """
        pass


class Normal(Initializator):
    """
    Normal initializer for neural network weights.
    """

    @staticmethod
    def _initialize(
        shape: tuple[int, ...], std: float, *, rng: Generator | None = None
    ) -> Tensor:
        """
        Initialize the weights for a neural network.

        Args:
            shape (tuple[int, ...]):
                The shape of the weight matrix.
            std (float):
                The standard deviation of the normal distribution used for initialization.
            rng (Generator or None, optional):
                A random number generator instance. If None, the default random generator is used.

        Returns:
            Tensor: A list containing the weight matrices for each layer.
        """
        rng = np.random.default_rng(rng)

        MEAN = 0

        return Tensor(rng.normal(MEAN, std, shape), requires_grad=True)


class Uniform(Initializator):
    """
    Uniform initializer for neural network weights.
    """

    @staticmethod
    def _initialize(
        shape: tuple[int, ...],
        bound: float,
        *,
        rng: Generator | None = None,
    ) -> Tensor:
        """
        Initialize the weights for a neural network.
        Args:
            shape (tuple[int, ...]):
                The shape of the weight matrix.
            rng (Generator or None, optional):
                A random number generator instance. If None, the default random generator is used.
        Returns:
            list[Tensor]: A list containing the weight matrices for each layer.
        """
        rng = np.random.default_rng(rng)

        return Tensor(rng.uniform(-bound, bound, shape), requires_grad=True)
