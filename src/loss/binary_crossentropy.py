import numpy as np

from .loss import Loss
from src.core import Tensor
import src.constants as c
from ..core.tensor import op
from src.encode import Encoder, BinaryEncoder


class BinaryCrossentropy(Loss):
    """
    BinaryCrossentropy loss class for binary classification tasks.
    """

    def __call__(
        self,
        expected: Tensor[np.floating],
        predicted: Tensor[np.floating],
    ) -> Tensor[np.floating]:
        if isinstance(expected, Tensor):
            if expected.shape != predicted.shape:
                raise ValueError(
                    f"Shape mismatch: expected {expected.shape}, got {predicted.shape}"
                )

        predicted.data = np.clip(predicted.data, c.EPSILON, 1 - c.EPSILON)

        sum_1 = expected * op.log(predicted)
        sum_2 = (1 - expected) * op.log(1 - predicted)

        if np.allclose(sum_1, 0):
            ce_loss = -sum_2
        elif np.allclose(sum_2, 0):
            ce_loss = -sum_1
        else:
            ce_loss = -(sum_1 + sum_2)

        return ce_loss

    @staticmethod
    def encoder() -> type[Encoder]:
        return BinaryEncoder
