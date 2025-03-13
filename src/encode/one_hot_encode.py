from typing import Final

import numpy as np

from .encoder import Encoder
from src.tensor import Tensor, op


class OneHotEncoder[T](Encoder):
    """
    OneHotEncoder is a class that encodes categorical labels into one-hot encoded vectors.
    """

    __slots__ = "_label_to_index"

    _label_to_index: Final[dict[T, int]]

    def __init__(self, labels: tuple[T, ...]):
        """
        Initializes the one-hot encoder with the given labels.
        Args:
            labels (tuple[T, ...]): A tuple of class labels.
        """
        super().__init__(labels)
        self._label_to_index = {c: i for i, c in enumerate(labels)}

    def encode(self, labels: Tensor) -> Tensor[np.floating]:
        """
        Encodes a given label into a one-hot encoded numpy array.

        Args:
            label (Tensor): Tensor of labels to be encoded.
        Returns:
            Tensor[np.floating]: A one-hot encoded numpy array representing the label.
        Raises:
            ValueError: If the label is not found in the predefined labels.
        """
        index = [self._label_to_index.get(label.item()) for label in labels]

        if None in index:
            raise ValueError("Label not found in predefined labels.")

        out = op.zeros((len(labels), len(self.labels)))
        out[range(len(labels)), index] = 1
        return out

    def decode(self, encoded: Tensor[np.floating] | Tensor[np.integer]) -> T:
        """
        Decodes a one-hot encoded numpy array to its corresponding class label.
        The encoded can also be a probabilities vector.

        Args:
            encoded (Tensor[np.floating] | Tensor[np.integer]): A one-hot encoded numpy array.
        Returns:
            T: The class label corresponding to the highest value in the encoded array.
        """
        return self.labels[int(op.argmax(encoded).item())]
