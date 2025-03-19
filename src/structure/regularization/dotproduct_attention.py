from typing import Any

from src.constants import EPSILON
from src.tensor import T, Tensor, op

from ..layer import Layer
from .dropout import Dropout


class DotProductAttention(Layer):
    """DotProductAttention layer."""

    __slots__ = "dropout"

    def __init__(self, dropout: float, rng: Any = None) -> None:
        """
        Initializes the DotProductAttention layer.
        Args:
            dropout: The dropout rate.
            rng: The random number generator.
        """
        self.dropout = Dropout(dropout, rng)

    def __call__(
        self, queries: Tensor[T], keys: Tensor[T], values: Tensor[T], attention_mask: Tensor | None = None
    ) -> Tensor[T]:
        scores = (queries @ op.transpose(keys, axes=(1, 2))) / op.sqrt(queries.shape[-1])

        if attention_mask is not None:
            attention_weights = scores + attention_mask * EPSILON

        return self.dropout(op.softmax(attention_weights)) @ values
