import numpy as np
from numpy.random import Generator

from .initialization import Uniform, Normal
from core import Tensor


class HeUniform(Uniform):
    """
    He Uniform initializer for neural network weights.
    """

    def initialize(
        self, shape: tuple[int, ...], *, rng: Generator | None = None
    ) -> list[Tensor]:
        in_features = np.prod(shape[1:])

        bound = self.gain * (np.sqrt(3 / in_features))

        return self._initialize(shape, bound, rng=rng)


class HeNormal(Normal):
    """
    He Normal initializer for neural network weights.
    """

    def initialize(
        self, shape: tuple[int, ...], *, rng: Generator | None = None
    ) -> list[Tensor]:
        in_features = np.prod(shape[1:])

        std = self.gain * (np.sqrt(2 / in_features))

        return self._initialize(shape, std, rng=rng)
