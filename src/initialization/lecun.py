import numpy as np
from numpy.random import Generator

from .initialization import Uniform, Normal
from src.tensor import Tensor


class LeCunUniform(Uniform):
    """LeCun Uniform initializer for neural network weights."""

    def initialize(
        self, shape: tuple[int, ...], *, requires_grad: bool = False, rng: Generator | None = None
    ) -> Tensor:
        in_features = np.prod(shape[1:])

        bound = self.gain * np.sqrt(3 / in_features)

        return self._initialize(shape, bound, requires_grad=requires_grad, rng=rng)


class LeCunNormal(Normal):
    """LeCun Normal initializer for neural network weights."""

    def initialize(
        self, shape: tuple[int, ...], *, requires_grad: bool = False, rng: Generator | None = None
    ) -> Tensor:
        in_features = np.prod(shape[1:])

        std = self.gain * np.sqrt(1 / in_features)

        return self._initialize(shape, std, requires_grad=requires_grad, rng=rng)
