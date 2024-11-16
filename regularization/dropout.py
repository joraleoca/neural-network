from dataclasses import dataclass


import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class Dropout:
    p: float = 0.1

    def __post_init__(self):
        """
        Post-initialization method to validate the dropout probability.

        Raises:
            ValueError: If `p` is not between 0 and 1.
        """
        if not (0 <= self.p <= 1):
            raise ValueError(
                f"The probability of dropout must be between 0 and 1. Got {self.p}."
            )

    def drop(self, layer_output: NDArray[np.floating]) -> None:
        """
        Apply dropout regularization to the given layer output in-place.
        Parameters:
            layer_output (NDArray[np.floating]): The output of the layer to which dropout will be applied.
        """
        rng = np.random.default_rng()

        mask = rng.binomial(1, 1 - self.p, size=layer_output.shape) / (1 - self.p)
        layer_output[:] *= mask
