from typing import Any

import numpy as np
from numpy.typing import NDArray

from .loss import Loss
from core.constants import EPSILON


class CategoricalCrossentropy(Loss):
    @staticmethod
    def loss(
        expected: NDArray[np.floating[Any]],
        predicted: NDArray[np.floating[Any]],
    ) -> np.floating[Any]:
        """
        Compute the categorical cross-entropy loss function with L2 regularization (weight decay).
        """
        if expected.shape != predicted.shape:
            raise ValueError(
                f"Shape mismatch: expected {expected.shape}, got {predicted.shape}"
            )

        clipped_predicted = np.clip(predicted, EPSILON, 1 - EPSILON)

        # Apply label smoothing to the expected output
        smoothing_factor = 0.1
        smoothed_expected = expected * (1 - smoothing_factor) + (
            smoothing_factor / len(expected)
        )

        # Calculate cross-entropy loss
        ce_loss = -np.sum(smoothed_expected * np.log(clipped_predicted))

        return ce_loss

    @staticmethod
    def gradient(
        expected: NDArray[np.floating[Any]],
        predicted: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """
        Compute the gradient of the categorical cross-entropy loss function.
        The weight decay gradient will be handled in the optimizer.
        """
        if expected.shape != predicted.shape:
            raise ValueError(
                f"Shape mismatch: expected {expected.shape}, got {predicted.shape}"
            )

        clipped_predicted = np.clip(predicted, EPSILON, 1 - EPSILON)

        # Apply label smoothing to the expected output
        smoothing_factor = 0.1
        smoothed_expected = expected * (1 - smoothing_factor) + (
            smoothing_factor / len(expected)
        )

        return clipped_predicted - smoothed_expected
