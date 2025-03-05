from typing import ClassVar, Any

import numpy as np
from numpy.random import Generator

from .trainable import Trainable
from src.core import Tensor, op
from src.initialization import Initializer
import src.constants as c


class Dense(Trainable):
    """Dense layer in a neural network."""

    __slots__ = "_in_features", "_out_features"

    _in_features: int
    _out_features: int

    required_fields: ClassVar[tuple[str, ...]] = (c.WEIGHT_PREFIX, c.BIAS_PREFIX)

    def __init__(
        self,
        features: int | tuple[int, int],
        initializer: Initializer | None = None,
        *,
        rng: Generator | None = None,
    ):
        """
        Initializes a new layer in the neural network.
        Args:
            features (int | tuple[int, int]):
                If int, the number of nodes in the layer, the in features are inferred from the first call to forward.
                If tuple, the number of (in features, out features)
            initializer (Initializer | None): The initializer for the weights of this layer. If None, the weights are not initialized.
            rng (Generator | None): A random number generator instance for initializing weights.
        Raises:
            ValueError: If any features is incorrect.
        """
        super().__init__(initializer, rng=rng)

        if isinstance(features, int):
            if features <= 0:
                raise ValueError(
                    f"The layer must have positive out features. Got {features}"
                )

            self._out_features = features
            self._in_features = -1
        elif isinstance(features, tuple):
            if len(features) != 2:
                raise ValueError(
                    f"The features must have len in and out features only. Got {features}."
                )
            for f in features:
                if f <= 0:
                    raise ValueError(
                        f"The layer must have positive features. Got {features}"
                    )

            self._in_features, self._out_features = features

            if self._initializer:
                self._initialize_weights()

        self.biases = op.zeros((1, self._out_features))
        self._requires_grad = False

    def __call__(self, data: Tensor[np.floating]) -> Tensor[np.floating]:
        if self._in_features == -1:
            self._in_features = data.shape[-1]
        if not hasattr(self, "weights"):
            self._initialize_weights()

        if data.shape[-1] != self._in_features:
            raise ValueError(
                f"Data has {data.shape[-1]} features but the layer has {self._in_features} features."
            )

        return (data @ self.weights) + self.biases 

    def _initialize_weights(self) -> None:
        """Initializes the weights of the layer."""
        assert self.requires_grad is not None, (
            "Requires grad cannot be None when initializing weights."
        )

        assert self._in_features is not None, (
            "Input features cannot be None when initializing weights."
        )
        assert self._initializer is not None, (
            "Initializer cannot be None when initializing weights."
        )

        self.weights = self._initializer.initialize(
            (self._in_features, self._out_features),
            requires_grad=self.requires_grad,
            rng=self.rng,
        )

        self._initializer = None

    @property
    def input_dim(self) -> int:
        """Returns the number of input features of the layer."""
        return self._in_features

    @property
    def output_dim(self) -> int:
        """Returns the number of output features of the layer."""
        return self._out_features

    def data_to_store(self) -> dict[str, Any]:
        return {
            c.WEIGHT_PREFIX: self.weights or None,
            c.BIAS_PREFIX: self.biases or None,
        }

    @staticmethod
    def from_data(data: dict[str, Any]) -> "Dense":
        weights = data[c.WEIGHT_PREFIX]
        in_features, out_features = weights.shape

        layer = Dense(out_features)

        layer._in_features = in_features
        layer.weights = Tensor(weights)
        layer.biases = Tensor(data[c.BIAS_PREFIX])

        return layer
