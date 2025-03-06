import numpy as np

from .loss import Loss
from src.core import Tensor, op
from src.encode import Encoder, OneHotEncoder


class CategoricalCrossentropy(Loss):
    """
    Categorical Crossentropy loss.
    """

    def __call__(
        self,
        expected: Tensor[np.floating],
        predicted: Tensor[np.floating],
    ) -> Tensor[np.floating]:
        # TODO: Fix the cce function to avoid this workaround due to the current implementation of the autograd.
        return op.cce(predicted, expected)

    @staticmethod
    def encoder() -> type[Encoder]:
        return OneHotEncoder
