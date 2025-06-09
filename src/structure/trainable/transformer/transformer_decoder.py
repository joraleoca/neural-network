import numpy as np

from src.tensor import Tensor

from ..trainable import Trainable
from ..batchnorm import BatchNorm
from ..multihead_attention import MultiHeadAttention
from ..positionwise_ffn import PositionWiseFFN


class TransformerDecoderBlock(Trainable):
    """TransformerDecoderBlock"""

    __slots__ = "features", "_self_attention", "_cross_attention", "_pw_ffn", "_norm1", "_norm2", "_norm3"

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

        self._self_attention = MultiHeadAttention(features, num_heads, dropout)
        self._cross_attention = MultiHeadAttention(features, num_heads, dropout)
        self._pw_ffn = PositionWiseFFN(features_feedforward, features)

        self._norm1 = BatchNorm(features, 2)
        self._norm2 = BatchNorm(features, 2)
        self._norm3 = BatchNorm(features, 2)

    def __call__(
        self,
        decoder_input: Tensor[np.floating],
        encoder_memory: Tensor[np.floating],
        attn_mask_decoder: Tensor | None = None,
        attn_mask_encoder: Tensor | None = None,
    ) -> Tensor[np.floating]:
        x = self._norm1(decoder_input)
        decoder_input = decoder_input + self._self_attention(x, x, x, attn_mask_decoder)

        x = self._norm2(decoder_input)
        decoder_input = decoder_input + self._cross_attention(x, encoder_memory, encoder_memory, attn_mask_encoder)

        x = self._norm3(decoder_input)
        decoder_input = decoder_input + self._pw_ffn(x)

        return decoder_input


class TransformerDecoder(Trainable):
    """TransformerDecoder layer"""

    __slots__ = "features", "blocks"

    def __init__(
        self,
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

    def __call__(
        self,
        decoder_input: Tensor[np.floating],
        encoder_memory: Tensor[np.floating],
        attn_mask_decoder: Tensor | None = None,
        attn_mask_encoder: Tensor | None = None,
    ) -> Tensor[np.floating]:
        for block in self.blocks:
            decoder_input = block(decoder_input, encoder_memory, attn_mask_decoder, attn_mask_encoder)

        return decoder_input
