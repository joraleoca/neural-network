from src.tensor import T, Tensor

from ..layer import Layer
from src.initialization import Initializer, XavierUniform


class Embedding(Layer):
    """Embedding layer."""

    __slots__ = "weights"

    def __init__(self, num_hiddens: int, vocab_size: int, initializer: Initializer = XavierUniform()) -> None:
        """
        Initializes the Embedding class.

        Args:
            num_hiddens (int): The number of hidden features.
            vocab_size (int): The vocabulary size.
            initializer (Initializer): Initializer for the weights.
        """
        if num_hiddens < 0:
            raise ValueError(f"Number of hidden features must be non-negative. Got {num_hiddens}")
        if vocab_size < 0:
            raise ValueError(f"Vocabulary size must be non-negative. Got {vocab_size}")

        self.weights = initializer.initialize((vocab_size, num_hiddens))

    def __call__(self, data: Tensor[T]) -> Tensor[T]:
        return self.weights[data.data]  # TODO: Check why tensor directly does not works

    def parameters(self) -> list[Tensor]:
        return [self.weights]
