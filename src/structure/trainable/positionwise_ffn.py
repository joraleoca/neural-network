import numpy as np

from ..trainable import Trainable
from .dense import Dense
from src.tensor import Tensor, op


class PositionWiseFFN(Trainable):
    """PositionWiseFFN layer"""

    __slots__ = "dense1", "dense2"

    def __init__(self, features: int, out_features: int) -> None:
        self.dense1 = Dense(features)
        self.dense2 = Dense(out_features)

    def __call__(self, data: Tensor[np.floating]) -> Tensor[np.floating]:
        return self.dense2(op.relu(self.dense1(data)))

    def parameters(self) -> list[Tensor]:
        return self.dense1.parameters() + self.dense2.parameters()
