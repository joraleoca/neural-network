import numpy as np

from .pool import Pool
from src.tensor import Tensor, op
from src.constants import EPSILON


class MaxPool(Pool):
    def __call__(self, data: Tensor[np.floating]) -> Tensor[np.floating]:
        data = self._pad(data, const_val=data.dtype.type(-EPSILON))

        windows = self._windows(data)

        return op.max(windows, axis=(-1, -2))
