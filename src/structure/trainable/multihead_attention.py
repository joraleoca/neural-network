from typing import Any

import numpy as np

from src.tensor import Tensor, op
from src.initialization import Initializer, XavierUniform

from .trainable import Trainable
from .dense import Dense


class MultiHeadAttention(Trainable):
    """MultiHeadAttention layer in a neural network."""

    __slots__ = "hidden_features", "num_heads", "W_o", "W_q", "W_k", "W_v", "dropout_p"

    def __init__(
        self,
        hidden_features: int,
        num_heads: int,
        dropout_p: float,
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
        super().__init__(initializer, rng=rng)

        self.hidden_features = hidden_features
        self.num_heads = num_heads

        self.W_o = Dense(hidden_features, initializer, rng=rng)
        self.W_k = Dense(hidden_features, initializer, rng=rng)
        self.W_q = Dense(hidden_features, initializer, rng=rng)
        self.W_v = Dense(hidden_features, initializer, rng=rng)
        self.dropout_p = dropout_p

    def __call__(
        self,
        queries: Tensor[np.floating],
        keys: Tensor[np.floating],
        values: Tensor[np.floating],
        valid_lens: Tensor[np.floating] | None = None,
    ) -> Tensor[np.floating]:
        queries = self._transpose_qkv(self.W_q(queries))
        keys = self._transpose_qkv(self.W_k(keys))
        values = self._transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            valid_lens = op.repeat(valid_lens, self.num_heads, axis=0)

        out = op.dotproduct_attention(queries, keys, values, valid_lens, self.dropout_p, rng=self.rng)

        # Reverse the _transpose_qkv operation
        out = out.reshape((-1, self.num_heads, out.shape[1], out.shape[2]))
        out = op.transpose(out, axes=(0, 2, 1, 3))
        out = out.reshape((out.shape[0], out.shape[1], -1))

        return self.W_o(out)

    def _transpose_qkv(self, arr: Tensor[np.floating]) -> Tensor[np.floating]:
        """
        Receives an arr of shape (batch size, num of queries, num hiddens) and returns
        the same arr with shape (batch size * num of heads, num queries, num hiddens / num of heads)
        """
        arr = arr.reshape(arr.shape[0:2] + (self.num_heads, -1))
        arr = op.transpose(arr, axes=(0, 2, 1, 3))
        return arr.reshape((-1,) + arr.shape[2:])

    def parameters(self) -> list[Tensor]:
        """
        Returns the parameters of the layer
        Returns:
            list[Tensor]: The parameters of the layer.
        """
        return self.W_o.parameters() + self.W_k.parameters() + self.W_q.parameters() + self.W_v.parameters()
