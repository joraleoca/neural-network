import numpy as np
from numpy.random import Generator

from .initialization import Uniform, Normal
from core import Tensor


class XavierUniform(Uniform):
    """
    Xavier Uniform initializer for neural network weights.
    """

    def initialize(
        self, network_structure: list[int], *, rng: Generator | None = None
    ) -> list[Tensor]:
        bound = self.gain * (
            np.sqrt(6 / (network_structure[0] + network_structure[-1]))
        )

        return self._initialize(network_structure, bound, rng=rng)


class XavierNormal(Normal):
    """
    Xavier Normal initializer for neural network weights.
    """

    def initialize(
        self, network_structure: list[int], *, rng: Generator | None = None
    ) -> list[Tensor]:
        std = self.gain * (np.sqrt(2 / (network_structure[0] + network_structure[-1])))

        return self._initialize(network_structure, std, rng=rng)
