from ..layer import Layer
from src.tensor import Tensor, T


class Flatten(Layer):
    """Flatten layer."""

    def __call__(self, data: Tensor[T]) -> Tensor[T]:
        return data.reshape((data.shape[0], -1))
