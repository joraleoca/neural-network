from typing import Any

import numpy as np

from src.tensor import Tensor, op
from src.initialization import Initializer, XavierUniform

from .trainable import Trainable


class Dense(Trainable):
    """Dense layer in a neural network."""

    __slots__ = "_in_features", "_out_features"

    _in_features: int
    _out_features: int

    def __init__(
        self,
        features: int | tuple[int, int],
        initializer: Initializer = XavierUniform(),
        *,
        rng: Any = None,
    ):
        """
        Initializes a new layer in the neural network.
        Args:
            features (int | tuple[int, int]):
                If int, the number of nodes in the layer, the in features are inferred from the first call to forward.
                If tuple, the number of (in features, out features)
            initializer (Initializer | None): The initializer for the weights of this layer. If None, the weights are not initialized.
            rng (Any): A random number generator instance for initializing weights.
        Raises:
            ValueError: If any features is incorrect.
        """
        super().__init__(initializer, rng=rng)

        if isinstance(features, int):
            if features <= 0:
                raise ValueError(f"The layer must have positive out features. Got {features}")

            self._out_features = features
            self._in_features = -1
        elif isinstance(features, tuple):
            if len(features) != 2:
                raise ValueError(f"The features must have in and out features only. Got {features}.")
            if any(f <= 0 for f in features):
                raise ValueError(f"The layer must have positive features. Got {features}")

            self._in_features, self._out_features = features
            self._initialize_weights()

        self.biases.set_data(op.zeros((1, self._out_features)))

    def __call__(self, data: Tensor[np.floating]) -> Tensor[np.floating]:
        if self._initializer is not None:
            self._in_features = data.shape[-1]
            self._initialize_weights()

        if data.shape[-1] != self._in_features:
            raise ValueError(f"Data has {data.shape[-1]} features but the layer has {self._in_features} features.")

        return (data @ self.weights) + self.biases

    def _initialize_weights(self) -> None:
        """Initializes the weights of the layer."""
        assert self._in_features > 0, (
            f"Input features have to be valid when initializing weights. Got {self._in_features}"
        )
        assert self._initializer is not None, "Initializer cannot be None when initializing weights."

        self.weights.set_data(
            self._initializer.initialize(
                (self._in_features, self._out_features),
                requires_grad=self.requires_grad,
                rng=self.rng,
            )
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
