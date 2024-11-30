from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator

from core import Tensor


class Initializator(ABC):
    """
    Initializator is an abstract base class for initializing neural network weights.
    """

    gain: float

    def __init__(self, gain: float = 1):
        self.gain = gain

    @abstractmethod
    def initialize(
        self, network_structure: list[int], *, rng: Generator | None = None
    ) -> list[Tensor]:
        """
        Initializes the neural network with the given structure.

        Args:
            network_structure (list[int]): A list of integers where each integer represents
                                       the number of neurons in each layer of the network.
            rng (Generator | None, optional): A random number generator instance. If None,
                                          the default random generator will be used.

        Returns:
            list[Tensor]: A list containing the weight matrices for each layer.
        """
        pass


class Normal(Initializator):
    """
    Normal initializer for neural network weights.
    """

    @staticmethod
    def _initialize(
        network_structure: list[int], std: float, *, rng: Generator | None = None
    ) -> list[Tensor]:
        """
        Initialize the weights for a neural network.
        Args:
            network_structure (list[int]):
                A list containing the number of neurons in each layer of the network.
            std (float):
                The standard deviation of the normal distribution used for initialization.
            rng (Generator or None, optional):
                A random number generator instance. If None, the default random generator is used.
        Returns:
            list[Tensor]: A list containing the weight matrices for each layer.
        """
        rng = np.random.default_rng(rng)

        MEAN = 0

        weights = [
            Tensor(
                rng.normal(MEAN, std, (network_structure[i + 1], network_structure[i])),
                requires_grad=True,
            )
            for i in range(len(network_structure) - 1)
        ]

        return weights


class Uniform(Initializator):
    """
    Uniform initializer for neural network weights.
    """

    @staticmethod
    def _initialize(
        network_structure: list[int],
        bound: float,
        *,
        rng: Generator | None = None,
    ) -> list[Tensor]:
        """
        Initialize the weights for a neural network.
        Args:
            network_structure (list[int]):
                A list containing the number of neurons in each layer of the network.
            rng (Generator or None, optional):
                A random number generator instance. If None, the default random generator is used.
        Returns:
            list[Tensor]: A list containing the weight matrices for each layer.
        """
        rng = np.random.default_rng(rng)

        weights = [
            Tensor(
                rng.uniform(
                    -bound, bound, (network_structure[i + 1], network_structure[i])
                ),
                requires_grad=True,
            )
            for i in range(len(network_structure) - 1)
        ]

        return weights
