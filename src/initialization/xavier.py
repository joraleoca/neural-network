import numpy as np
from numpy.random import Generator

from .initialization import Uniform, Normal
from src.core import Tensor


class XavierUniform(Uniform):
    """
    Xavier Uniform initializer for neural network weights.
    """

    def initialize(
        self, shape: tuple[int, ...], *, requires_grad: bool = False, rng: Generator | None = None
    ) -> Tensor:
        in_features = np.prod(shape[1:])
        out_features = shape[0]

        bound = self.gain * (np.sqrt(6 / (in_features + out_features)))

        return self._initialize(shape, bound, requires_grad=requires_grad, rng=rng)


class XavierNormal(Normal):
    """
    Xavier Normal initializer for neural network weights.
    """

    def initialize(
        self, shape: tuple[int, ...], *, requires_grad: bool = False, rng: Generator | None = None
    ) -> Tensor:
        in_features = np.prod(shape[1:])
        out_features = shape[0]

        std = self.gain * (np.sqrt(2 / (in_features + out_features)))

        return self._initialize(shape, std, requires_grad=requires_grad, rng=rng)
