import numpy as np

from src.tensor import Tensor

from ..batchnorm import BatchNorm
from ..multihead_attention import MultiHeadAttention
from ..positionwise_ffn import PositionWiseFFN
from ..trainable import Trainable


class TransformerEncoderBlock(Trainable):
    """TransformerEncoderBlock"""

    __slots__ = "features", "_attention", "_pw_ffn", "_norm1", "_norm2"

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
        self._attention = MultiHeadAttention(features, num_heads, dropout)
        self._pw_ffn = PositionWiseFFN(features_feedforward, features)
        self._norm1 = BatchNorm(features, 2)
        self._norm2 = BatchNorm(features, 2)

    def __call__(self, data: Tensor[np.floating], attn_mask: Tensor[np.bool] | None = None) -> Tensor[np.floating]:
        norm_data = self._norm1(data)
        data = data + self._attention(norm_data, norm_data, norm_data, attn_mask)

        norm_data = self._norm2(data)
        data = data + self._pw_ffn(norm_data)

        return data


class TransformerEncoder(Trainable):
    """TransformerEncoder layer"""

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
            TransformerEncoderBlock(features, num_heads, features_feedforward, dropout) for _ in range(num_blocks)
        ]

    def __call__(self, data: Tensor[np.floating], attn_mask: Tensor | None = None) -> Tensor[np.floating]:
        for block in self.blocks:
            data = block(data, attn_mask)

        return data
