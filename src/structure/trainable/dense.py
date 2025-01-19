from typing import ClassVar, Any

import numpy as np
from numpy.random import Generator

from .trainable import Trainable
from src.core import Tensor, op
from src.activation import ActivationFunction, activation_from_name
from src.initialization import Initializer
import src.constants as c


class Dense(Trainable):
    """Dense layer in a neural network."""

    __slots__ = [
        "_in_features",
        "_out_features",
    ]

    _in_features: int
    _out_features: int

    required_fields: ClassVar[set[str]] = (c.WEIGHT_PREFIX, c.BIAS_PREFIX, c.ACTIVATION_PREFIX)

    def __init__(
        self,
        features: int | tuple[int, int],
        activation_function: ActivationFunction | None = None,
        weights_initializer: Initializer | None = None,
        *,
        rng: Generator | None = None,
    ):
        """
        Initializes a new layer in the neural network.
        Args:
            features (int | tuple[int, int]):
                If int, the number of nodes in the layer, the in features are inferred from the first call to forward.
                If tuple, the number of (in features, out features)
            activation_function (ActivationFunction | None): The activation function to be used by this layer.
            weights_initializer (Initializer | None): The initializer for the weights of this layer. If None, the weights are not initialized.
            rng (Generator | None): A random number generator instance for initializing weights.
        Raises:
            ValueError: If any features is incorrect.
        """
        super().__init__()

        if isinstance(features, int):
            if features <= 0:
                raise ValueError(
                    f"The layer must have positive out features. Got {features}"
                )

            self._out_features = features
            self._in_features = -1
            self._initializer = weights_initializer
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
            self._initializer = weights_initializer

            if self._initializer:
                self._initialize_weights(rng=rng)

        self.activation_function = activation_function
        self.biases = op.zeros((1, self._out_features))
        self._requires_grad = False
        self.rng = rng

    @property
    def input_dim(self) -> int:
        """Returns the number of input features of the layer."""
        return self._in_features

    @property
    def output_dim(self) -> int:
        """Returns the number of output features of the layer."""
        return self._out_features

    def forward(self, data: Tensor[np.floating]) -> Tensor[np.floating]:
        if self._in_features == -1:
            self._in_features = data.shape[-1]
        if not hasattr(self, "weights"):
            self._initialize_weights(requires_grad=self.requires_grad, rng=self.rng)

        if data.shape[-1] != self._in_features:
            raise ValueError(
                f"Data has {data.shape[-1]} features but the layer has {self._in_features} features."
            )

        forward_output = (data @ self.weights) + self.biases

        if self.activation_function:
            forward_output = self.activation_function(forward_output)

        return forward_output

    def _initialize_weights(self, *, requires_grad: bool = False, rng: Generator | None = None) -> None:
        """Initializes the weights of the layer."""
        assert self._in_features is not None, "Input features cannot be None when initializing weights."
        assert self._initializer is not None, "Initializer cannot be None when initializing weights."

        self.weights = self._initializer.initialize(
            (self._in_features, self._out_features),
            requires_grad=requires_grad,
            rng=rng,
        )

        self._initializer = None

    def data_to_store(self) -> dict[str, Any]:
        return {
            c.WEIGHT_PREFIX: self.weights or None,
            c.BIAS_PREFIX: self.biases or None,
            c.ACTIVATION_PREFIX: self.activation_function.__class__.__name__ or None,
        }

    @staticmethod
    def from_data(data: dict[str, Any]) -> "Dense":
        weights = data[c.WEIGHT_PREFIX]
        in_features, out_features = weights.shape

        layer = Dense(out_features)

        layer._in_features = in_features
        layer.weights = Tensor(weights)
        layer.biases = Tensor(data[c.BIAS_PREFIX])
        layer.activation_function = activation_from_name(data[c.ACTIVATION_PREFIX].item())()

        return layer

