from typing import Any

import numpy as np

from .loss import Loss
from core.constants import EPSILON
from encode import Encoder, BinaryEncoder


class BinaryCrossentropy(Loss):
    @staticmethod
    def loss(
        expected: np.floating[Any],
        predicted: np.floating[Any],
    ) -> np.floating[Any]:
        clipped_predicted = np.clip(predicted, EPSILON, 1 - EPSILON)

        ce_loss = -(
            expected * np.log(clipped_predicted)
            + (1 - expected) * np.log(1 - clipped_predicted)
        )

        return ce_loss

    @staticmethod
    def gradient(
        expected: np.floating[Any],
        predicted: np.floating[Any],
    ) -> np.floating[Any]:
        clipped_predicted = np.clip(predicted, EPSILON, 1 - EPSILON)

        return clipped_predicted - expected

    @staticmethod
    def encoder() -> type[Encoder]:
        return BinaryEncoder
