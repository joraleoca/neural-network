from typing import Final

import numpy as np
from numpy.typing import NDArray

from .encoder import Encoder


class OneHotEncoder(Encoder):
    """
    OneHotEncoder is a class that encodes categorical labels into one-hot encoded vectors.
    """

    classes: tuple[str, ...]
    _label_to_index: Final[dict[str, int]]

    def __init__(self, classes: tuple[str, ...]):
        """
        Initializes the one-hot encoder with the given classes.

        Args:
            classes (tuple[str, ...]): A tuple of class labels.
        """
        self.classes = classes
        self._label_to_index = {c: i for i, c in enumerate(classes)}

    def encode(self, label: str) -> NDArray[np.floating]:
        """
        Encodes a given label into a one-hot encoded numpy array.
        Args:
            label (str): The label to be encoded.
        Returns:
            NDArray[np.floating]: A one-hot encoded numpy array representing the label.
        Raises:
            ValueError: If the label is not found in the predefined classes.
        """
        if label not in self._label_to_index:
            raise ValueError("Label is not in classes")

        encode = np.zeros((len(self._label_to_index.keys()), 1), dtype=np.float64)
        encode[self._label_to_index[label]] = 1

        return encode

    def decode(
        self, encoded: NDArray[np.floating] | NDArray[np.integer]
    ) -> str:
        """
        Decodes a one-hot encoded numpy array to its corresponding class label.
        The encoded can also be a probabilities vector.
        Args:
            encoded (NDArray[np.floating] | NDArray[np.integer): A one-hot encoded numpy array.
        Returns:
            str: The class label corresponding to the highest value in the encoded array.
        """

        return self.classes[encoded.argmax()]
