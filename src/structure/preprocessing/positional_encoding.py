import numpy as np

from ..layer import Layer
from src.tensor import Tensor, T, op


class PositionalEncoding(Layer):
    """PositionalEncoding layer."""

    __slots__ = "P", "dropout_p"

    def __init__(self, num_hiddens: int, max_len: int = 1000, dropout_p: float = 0) -> None:
        """
        Initializes the PositionalEncoding class.

        Args:
            num_hiddens (int): The number of hidden features.
            max_len (int): The maximum length of the input sequence.
            dropout_p (float): The dropout probability.
        """

        if num_hiddens < 0:
            raise ValueError(f"Number of hidden features must be non-negative. Got {num_hiddens}")
        if max_len < 0:
            raise ValueError(f"Max len must be non-negative. Got {max_len}")

        i = np.arange(max_len).reshape(-1, 1)
        j = np.arange(num_hiddens).reshape(1, -1)

        p = i / (10000 ** (2 * j / num_hiddens))
        p[:, 0::2] = np.sin(p[:, 0::2])
        p[:, 1::2] = np.cos(p[:, 1::2])
        p = np.expand_dims(p, 0)

        self.P = Tensor(p)
        self.dropout_p = dropout_p

    def __call__(self, data: Tensor[T]) -> Tensor[T]:
        return op.dropout(data + self.P[:, : data.shape[1], :], self.dropout_p)
