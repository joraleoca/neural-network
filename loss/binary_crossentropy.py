import numpy as np

from .loss import Loss
from core import Tensor, op, constants as c
from encode import Encoder, BinaryEncoder


class BinaryCrossentropy(Loss):
    """
    BinaryCrossentropy loss class for binary classification tasks.
    """

    @staticmethod
    def __call__(
        expected: Tensor[np.floating] | np.integer | np.floating,
        predicted: Tensor[np.floating] | np.integer | np.floating,
    ) -> Tensor[np.floating]:
        if not isinstance(predicted, Tensor):
            predicted = Tensor(predicted, dtype=np.float32)

        if isinstance(expected, Tensor):
            if expected.shape != predicted.shape:
                raise ValueError(
                    f"Shape mismatch: expected {expected.shape}, got {predicted.shape}"
                )
            if predicted.shape != (1,):
                raise ValueError(f"Expected shape (1,), got {predicted.shape}")

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
