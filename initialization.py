"""
initialization.py
This module implements various weight initialization strategies for neural networks.
Each initializer is implemented as a concrete class inheriting from the abstract
Initializator base class.
    Initializator: Abstract base class defining the interface for weight initializers.
    Normal: Implementation of weight initialization using a normal distribution.
    Uniform: Implementation of weight initialization using a uniform distribution.
    HeUniform: Implementation of He Uniform initialization for neural network weights.
    HeNormal: Implementation of He Normal initialization for neural network weights.
    XavierUniform: Implementation of Xavier Uniform initialization for neural network weights.
    XavierNormal: Implementation of Xavier Normal initialization for neural network weights.
    >>> from initialization import HeNormal
    >>> # Create initializer instance
    >>> initializer = HeNormal(gain=1.0)
    >>> # Define network structure
    >>> network_structure = [3, 5, 2]
    >>> # Initialize weights
    >>> weights = initializer.initializate(network_structure)
    >>> for w in weights:
    >>>     print(w.shape)  # Output: (5, 3) and (2, 5)
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator


class Initializator(ABC):
    gain: float

    def __init__(self, gain: float = 1):
        self.gain = gain

    @abstractmethod
    def initializate(
        self, network_structure: list[int], *, rng: Generator | None = None
    ) -> list[NDArray]:
        """
        Initializes the neural network with the given structure.

        Args:
            network_structure (list[int]): A list of integers where each integer represents
                                       the number of neurons in each layer of the network.
            rng (Generator | None, optional): A random number generator instance. If None,
                                          the default random generator will be used.

        Returns:
            list[NDArray]: A list containing the weight matrices for each layer.
        """
        pass


class Normal(Initializator):
    """
    Normal initializer for neural network weights.
    """

    def _initializate(
        self, network_structure: list[int], std: float, *, rng: Generator | None = None
    ) -> list[NDArray]:
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
            list[NDArray]: A list containing the weight matrices for each layer.
        """
        rng = np.random.default_rng(rng)

        MEAN = 0

        weights = [
            rng.normal(MEAN, std, (network_structure[i + 1], network_structure[i]))
            for i in range(len(network_structure) - 1)
        ]

        return weights


class Uniform(Initializator):
    """
    Initialize the weights for a neural network.
    """

    def _initializate(
        self,
        network_structure: list[int],
        bound: float,
        *,
        rng: Generator | None = None,
    ) -> list[NDArray]:
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
            list[NDArray]: A list containing the weight matrices for each layer.
        """
        rng = np.random.default_rng(rng)

        weights = [
            rng.uniform(-bound, bound, (network_structure[i + 1], network_structure[i]))
            for i in range(len(network_structure) - 1)
        ]

        return weights


class HeUniform(Uniform):
    """
    He Uniform initializer for neural network weights.
    """

    def initializate(
        self, network_structure: list[int], *, rng: Generator | None = None
    ) -> list[NDArray]:
        bound = self.gain * (np.sqrt(3 / network_structure[0]))

        return self._initializate(network_structure, bound, rng=rng)


class HeNormal(Normal):
    """
    He Normal initializer for neural network weights.
    """

    def initializate(
        self, network_structure: list[int], *, rng: Generator | None = None
    ) -> list[NDArray]:
        std = self.gain * (np.sqrt(2 / network_structure[0]))

        return self._initializate(network_structure, std, rng=rng)


class XavierUniform(Uniform):
    """
    Xavier Uniform initializer for neural network weights.
    """

    def initializate(
        self, network_structure: list[int], *, rng: Generator | None = None
    ) -> list[NDArray]:
        bound = self.gain * (
            np.sqrt(6 / (network_structure[0] + network_structure[-1]))
        )

        return self._initializate(network_structure, bound, rng=rng)


class XavierNormal(Normal, Initializator):
    """
    Xavier Normal initializer for neural network weights.
    """

    def initializate(
        self, network_structure: list[int], *, rng: Generator | None = None
    ) -> list[NDArray]:
        std = self.gain * (np.sqrt(2 / (network_structure[0] + network_structure[-1])))

        return self._initializate(network_structure, std, rng=rng)
