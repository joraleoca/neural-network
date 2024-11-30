import numpy as np

from .loss import Loss
from core import Tensor, op, constants as c
from encode import Encoder, OneHotEncoder


class CategoricalCrossentropy(Loss):
    """
    Categorical Crossentropy loss.
    """

    __slots__ = ["smoothing_factor"]

    smoothing_factor: float

    def __init__(self, smoothing_factor: float = 0):
        if not 0 <= smoothing_factor <= 1:
            raise ValueError("Smoothing factor must be between 0 and 1 (inclusive)")
        self.smoothing_factor = smoothing_factor

    def __call__(
        self,
        expected: Tensor[np.floating],
        predicted: Tensor[np.floating],
    ) -> Tensor[np.floating]:
        if expected.shape != predicted.shape:
            raise ValueError(
                f"Shape mismatch: expected {expected.shape}, got {predicted.shape}"
            )

        # Clipping
        predicted.data = np.clip(predicted.data, c.EPSILON, 1 - c.EPSILON)
        expected = Tensor(
            np.clip(expected.data, c.EPSILON, 1 - c.EPSILON), dtype=expected.dtype
        )

        # Apply label smoothing to the expected output
        if self.smoothing_factor != 0:
            num_classes = expected.shape[0]
            expected = expected * (1 - self.smoothing_factor) + (
                self.smoothing_factor / num_classes
            )

        # Calculate cross-entropy loss
        ce_loss = -op.sum(expected * op.log(predicted), axis=-1)

        return ce_loss  # type: ignore

    @staticmethod
    def encoder() -> type[Encoder]:
        return OneHotEncoder
