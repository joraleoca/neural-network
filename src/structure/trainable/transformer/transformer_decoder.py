from itertools import chain

import numpy as np

from src.tensor import Tensor, op

from ...preprocessing import Embedding, PositionalEncoding
from ..batchnorm import BatchNorm
from ..dense import Dense
from ..multihead_attention import MultiHeadAttention
from ..positionwise_ffn import PositionWiseFFN
from ..trainable import Trainable


class TransformerDecoderBlock(Trainable):
    """TransformerDecoderBlock"""

    __slots__ = "features", "_self_attention", "_cross_attention", "_pw_ffn", "_norm1", "_norm2", "_norm3", "dropout"

    def __init__(
        self,
        features: int,
        num_heads: int,
        features_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        if features < 0:
            raise ValueError(f"Features must be non-negative. Got {features}")

        self.features = features
        self.dropout = dropout

        self._self_attention = MultiHeadAttention(features, num_heads, dropout)
        self._cross_attention = MultiHeadAttention(features, num_heads, dropout)
        self._pw_ffn = PositionWiseFFN(features_feedforward, features)

        self._norm1 = BatchNorm(features, 2)
        self._norm2 = BatchNorm(features, 2)
        self._norm3 = BatchNorm(features, 2)

    def __call__(
        self,
        data: Tensor[np.floating],
        state: Tensor[np.floating],
        valid_lens_data: Tensor | None = None,
        valid_lens_state: Tensor | None = None,
    ) -> Tensor[np.floating]:
        if valid_lens_state is None:
            valid_lens_state = op.repeat(Tensor.default_module.arange(1, state.shape[1] + 1), state.shape[2], axis=1)

        norm_state = self._norm1(state)
        state = state + self._self_attention(norm_state, norm_state, norm_state, valid_lens_state)

        if valid_lens_data is None:
            valid_lens_data = op.repeat(Tensor.default_module.arange(1, data.shape[1] + 1), data.shape[2], axis=1)

        norm_state = self._norm2(state)
        state = state + self._cross_attention(norm_state, data, data, valid_lens_data)

        norm_state = self._norm3(state)
        state = state + self._pw_ffn(norm_state)

        return state

    def parameters(self) -> list[Tensor]:
        return (
            self._self_attention.parameters()
            + self._cross_attention.parameters()
            + self._norm1.parameters()
            + self._norm2.parameters()
            + self._norm3.parameters()
            + self._pw_ffn.parameters()
        )


class TransformerDecoder(Trainable):
    """TransformerDecoder layer"""

    __slots__ = "features", "blocks", "pos_encoding", "embedding", "dense"

    def __init__(
        self,
        vocab_size: int,
        features: int,
        num_heads: int,
        features_feedforward: int,
        num_blocks: int,
        dropout: float = 0.1,
    ) -> None:
        self.features = features
        self.blocks = [
            TransformerDecoderBlock(features, num_heads, features_feedforward, dropout) for _ in range(num_blocks)
        ]
        self.pos_encoding = PositionalEncoding(features, dropout_p=dropout)
        self.embedding = Embedding(features, vocab_size)
        self.dense = Dense(features)

    def __call__(
        self, data: Tensor[np.floating], state: Tensor[np.floating], valid_lens: Tensor | None = None
    ) -> Tensor[np.floating]:
        data = self.pos_encoding((self.embedding(data) * op.sqrt(self.features)))

        for block in self.blocks:
            state = block(data, state, valid_lens)

        return self.dense(state)

    def parameters(self) -> list[Tensor]:
        return list(chain.from_iterable(block.parameters() for block in self.blocks)) + self.dense.parameters()
