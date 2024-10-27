from initialization.initialization import Uniform, Normal

import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator


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


class XavierNormal(Normal):
    """
    Xavier Normal initializer for neural network weights.
    """

    def initializate(
        self, network_structure: list[int], *, rng: Generator | None = None
    ) -> list[NDArray]:
        std = self.gain * (np.sqrt(2 / (network_structure[0] + network_structure[-1])))

        return self._initializate(network_structure, std, rng=rng)
