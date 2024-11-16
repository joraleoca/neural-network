

import numpy as np

from .loss import Loss
from core.constants import EPSILON
from encode import Encoder, BinaryEncoder


class BinaryCrossentropy(Loss):
    """
    BinaryCrossentropy loss class for binary classification tasks.
    """

    @staticmethod
    def loss(
        expected: np.floating,
        predicted: np.floating,
    ) -> np.floating:
        clipped_predicted = np.clip(predicted, EPSILON, 1 - EPSILON)

        ce_loss = -(
            expected * np.log(clipped_predicted)
            + (1 - expected) * np.log(1 - clipped_predicted)
        )

        return ce_loss

    @staticmethod
    def gradient(
        expected: np.floating,
        predicted: np.floating,
    ) -> np.floating:
        clipped_predicted = np.clip(predicted, EPSILON, 1 - EPSILON)

        return clipped_predicted - expected

    @staticmethod
    def encoder() -> type[Encoder]:
        return BinaryEncoder
