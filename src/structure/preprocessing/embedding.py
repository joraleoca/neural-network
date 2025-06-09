from src.tensor import Tensor, op

import cupy as cp

from ..layer import Layer
from ..parameter import Parameter
from src.initialization import Initializer, XavierUniform


class Embedding(Layer):
    """Embedding layer."""

    __slots__ = "weights"

    def __init__(self, features: int, vocab_size: int, initializer: Initializer = XavierUniform()) -> None:
        """
        Initializes the Embedding class.

        Args:
            features (int): The number of features.
            vocab_size (int): The vocabulary size.
            initializer (Initializer): Initializer for the weights.
        """
        if features < 0:
            raise ValueError(f"Number of hidden features must be non-negative. Got {features}")
        if vocab_size < 0:
            raise ValueError(f"Vocabulary size must be non-negative. Got {vocab_size}")

        self.weights = Parameter(initializer.initialize((vocab_size, features)))

    def __call__(self, data: Tensor[cp.integer]) -> Tensor:
        # TODO: Fix this workaround
        indices = data.data
        self.weights.requires_grad = data.requires_grad

        if indices.ndim > 1:
            out = op.concat([self.weights[i] for i in indices.ravel()]).reshape((*indices.shape, -1))
        else:
            out = self.weights[indices]

        return out
