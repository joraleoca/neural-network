from typing import Any

import numpy as np

from src.tensor import Tensor, op
from src.initialization import Initializer, XavierUniform

from ..activation import ActivationFunction, Tanh
from ..parameter import Parameter
from .trainable import Trainable


class Recurrent(Trainable):
    """Recurrent layer in a neural network."""

    __slots__ = "weights", "biases", "activation", "hidden_features", "_features", "_internal_weights"

    def __init__(
        self,
        features: int,
        hidden_features: int,
        activation: ActivationFunction = Tanh(),
        initializer: Initializer = XavierUniform(),
        *,
        rng: Any = None,
    ):
        """
        Initializes a new layer in the neural network.
        Args:
            features (int): The number of features in the layer.
            hidden_features (int): The number of hidden features in the layer.
            activation (ActivationFunction): The activation function for the layer.
            initializer (Initializer | None): The initializer for the weights of this layer. If None, the weights are not initialized.
            rng (Any): A random number generator instance for initializing weights.
        Raises:
            ValueError: If any features is incorrect.
        """
        super().__init__(rng=rng)

        self._features = features
        self.hidden_features = hidden_features
        self.activation = activation

        self.weights = Parameter(
            initializer.initialize((features, hidden_features), requires_grad=self.requires_grad, rng=rng)
        )
        self.biases = Parameter(op.zeros((1, hidden_features)))

        self._internal_weights = Parameter(
            initializer.initialize((hidden_features, hidden_features), requires_grad=self.requires_grad, rng=rng)
        )

    def __call__(self, data: Tensor[np.floating], state: Tensor[np.floating] | None = None) -> Tensor[np.floating]:
        batch_size, seq_len, features = data.shape

        if features != self._features:
            raise ValueError(f"Data has {features} features but the layer has {self._features} features.")

        if state is None:
            state = op.zeros((batch_size, self.hidden_features))

        out = []

        for i in range(seq_len):
            state = self.activation((data[:, i, :] @ self.weights) + self.biases + (state @ self._internal_weights))
            out.append(state)

        return op.stack(out, axis=1)
