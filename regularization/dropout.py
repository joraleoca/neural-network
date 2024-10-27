from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class Dropout:
    p: float = 0.1

    def __post_init__(self):
        if not (0 <= self.p <= 1):
            raise ValueError(
                f"The probability of dropout must be between 0 and 1. Got {self.p}."
            )

    def drop(self, layer_output: NDArray[np.floating[Any]]) -> None:
        """
        Apply dropout regularization to the given layer output in-place.
        Parameters:
            layer_output (NDArray[np.floating[Any]]): The output of the layer to which dropout will be applied.
        """
        mask = np.random.binomial(1, 1 - self.p, size=layer_output.shape) / (1 - self.p)
        layer_output[:] *= mask
