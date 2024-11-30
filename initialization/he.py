import numpy as np
from numpy.random import Generator

from .initialization import Uniform, Normal
from core import Tensor


class HeUniform(Uniform):
    """
    He Uniform initializer for neural network weights.
    """

    def initialize(
        self, network_structure: list[int], *, rng: Generator | None = None
    ) -> list[Tensor]:
        bound = self.gain * (np.sqrt(3 / network_structure[0]))

        return self._initialize(network_structure, bound, rng=rng)


class HeNormal(Normal):
    """
    He Normal initializer for neural network weights.
    """

    def initialize(
        self, network_structure: list[int], *, rng: Generator | None = None
    ) -> list[Tensor]:
        std = self.gain * (np.sqrt(2 / network_structure[0]))

        return self._initialize(network_structure, std, rng=rng)
