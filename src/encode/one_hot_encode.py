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
        out = []

        for label in labels.ravel():
            one_hot = op.zeros((len(self._label_to_index),))
            one_hot[self._label_to_index[label.item()]] = 1
            out.append(one_hot)

        return op.stack(out).reshape(labels.shape + (len(self._label_to_index),))

    def decode(self, encoded: Tensor[np.floating] | Tensor[np.integer]) -> Tensor:
        """
        Decodes a one-hot encoded numpy array to its corresponding class label.
        The encoded can also be a probabilities vector.

        Args:
            encoded (Tensor[np.floating] | Tensor[np.integer]): A one-hot encoded numpy array.
        Returns:
            T: The class label corresponding to the highest value in the encoded array.
        """
        index = op.argmax(encoded, axis=-1).ravel()
        out = Tensor([self.labels[i.item()] for i in index], dtype=type(self.labels[0]))
        return out.reshape(encoded.shape[:-1])
