from typing import Final

import numpy as np

from .encoder import Encoder
from src.core import Tensor
from ..core.tensor import op


class OneHotEncoder[T](Encoder):
    """
    OneHotEncoder is a class that encodes categorical labels into one-hot encoded vectors.
    """

    __slots__ = "_label_to_index"

    _label_to_index: Final[dict[T, int]]

    def __init__(self, classes: tuple[T, ...]):
        """
        Initializes the one-hot encoder with the given classes.
        Args:
            classes (tuple[T, ...]): A tuple of class labels.
        """
        super().__init__(classes)
        self._label_to_index = {c: i for i, c in enumerate(classes)}

    def encode(self, label: T) -> Tensor[np.floating]:
        """
        Encodes a given label into a one-hot encoded numpy array.

        Args:
            label (T): The label to be encoded.
        Returns:
            Tensor[np.floating]: A one-hot encoded numpy array representing the label.
        Raises:
            ValueError: If the label is not found in the predefined classes.
        """
        if label not in self._label_to_index:
            raise ValueError(f"Label is not in classes. Got {label}")

        encode = op.zeros((len(self._label_to_index.keys()),))
        encode[self._label_to_index[label]] = 1

        return encode

    def decode(self, encoded: Tensor[np.floating] | Tensor[np.integer]) -> T:
        """
        Decodes a one-hot encoded numpy array to its corresponding class label.
        The encoded can also be a probabilities vector.

        Args:
            encoded (Tensor[np.floating] | Tensor[np.integer]): A one-hot encoded numpy array.
        Returns:
            T: The class label corresponding to the highest value in the encoded array.
        """
        return self.classes[int(op.argmax(encoded).item())]
