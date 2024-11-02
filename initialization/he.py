from .initialization import Uniform, Normal

import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator


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
