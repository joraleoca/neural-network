import numpy as np
from numpy.random import Generator

from core import Tensor


def dropout(
    layer_output: Tensor[np.floating], *, p: float, rng: Generator | None = None
) -> Tensor[np.floating]:
    """
    Apply dropout regularization to the given layer output.

    Args:
        layer_output (Tensor[np.floating]): The output of the layer to which dropout will be applied.
        p (float): The probability of keeping a neuron active.
        rng (Generator | None): The random number generator to use.
    """
    if p == 0:
        return layer_output

    if p < 0 or p > 1:
        raise ValueError("The dropout probability must be between 0 and 1.")

    rng = np.random.default_rng(rng)

    mask = rng.binomial(1, 1 - p, size=layer_output.shape) / (1 - p)

    return layer_output * mask
