import numpy as np

from .loss import Loss
from src.tensor import Tensor, op
from src.encode import Encoder, OneHotEncoder


class CategoricalCrossentropy(Loss):
    """
    Categorical Crossentropy loss.
    """

    __slots__ = "ignore_token_id"

    def __init__(self, ignore_token_id: int | None = None) -> None:
        self.ignore_token_id = ignore_token_id

    def __call__(
        self,
        predicted: Tensor[np.floating],
        expected: Tensor[np.floating],
    ) -> Tensor[np.floating]:
        # TODO: Fix the cce function to avoid this workaround due to the current implementation of the autograd.
        return op.cce(predicted, expected, self.ignore_token_id)

    @staticmethod
    def encoder() -> type[Encoder]:
        return OneHotEncoder
