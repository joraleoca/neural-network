from typing import Any

import numpy as np

from src.tensor import Tensor, op
from src.initialization import Initializer, XavierUniform

from .trainable import Trainable
from .dense import Dense


class MultiHeadAttention(Trainable):
    """MultiHeadAttention layer in a neural network."""

    __slots__ = "num_heads", "W_o", "W_q", "W_k", "W_v", "dropout_p"

    def __init__(
        self,
        features: int,
        num_heads: int,
        dropout_p: float,
        initializer: Initializer = XavierUniform(),
        *,
        rng: Any = None,
    ):
        """
        Initializes a multi-head attention layer.
        Args:
            features (int): The number of features in the layer.
            num_heads (int): The number of heads in the layer.
            activation (ActivationFunction): The activation function for the layer.
            initializer (Initializer): The initializer for the weights of this layer.
            rng (Any): A random number generator instance for initializing weights.
        """
        super().__init__(rng=rng)

        if not features % num_heads == 0:
            raise ValueError(f"Features must be divisible by num_heads. Got {features=} and {num_heads=}")

        self.num_heads = num_heads

        dense_features = (features, features)
        self.W_o = Dense(dense_features, initializer, rng=rng)
        self.W_k = Dense(dense_features, initializer, rng=rng)
        self.W_q = Dense(dense_features, initializer, rng=rng)
        self.W_v = Dense(dense_features, initializer, rng=rng)
        self.dropout_p = dropout_p

    def __call__(
        self,
        queries: Tensor[np.floating],
        keys: Tensor[np.floating],
        values: Tensor[np.floating],
        attn_mask: Tensor[np.bool] | None = None,
    ) -> Tensor[np.floating]:
        """
        Args:
            queries, keys, values (Tensor[floating]): Tensors of shape (batch_size, seq_len, features)
            attn_mask (Tensor | None): Float mask broadcastable to (B*H, Tq, Tk), where masked positions are -inf.

        Returns:
            Output tensor of shape (batch_size, seq_len, features)
        """
        queries = self._transpose_qkv(self.W_q(queries))
        keys = self._transpose_qkv(self.W_k(keys))
        values = self._transpose_qkv(self.W_v(values))

        if attn_mask is not None:
            if attn_mask.shape[0] == queries.shape[0] // self.num_heads:
                attn_mask = op.repeat(attn_mask, self.num_heads, axis=0)

        out = op.dotproduct_attention(queries, keys, values, attn_mask, self.dropout_p, rng=self.rng)

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
        arr = arr.reshape((arr.shape[0], arr.shape[1], self.num_heads, -1))
        arr = op.transpose(arr, axes=(0, 2, 1, 3))
        return arr.reshape((-1, arr.shape[2], arr.shape[3]))

    @property
    def features(self) -> int:
        """
        Returns the number of features in the layer.
        Returns:
            int: The number of features in the layer.
        """
        return self.W_k.in_features
