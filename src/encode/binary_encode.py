import cupy as cp
import numpy as np

from .encoder import Encoder
from src.tensor import Tensor


class BinaryEncoder[T](Encoder):
    """
    BinaryEncoder is a class that encodes categorical labels into a binary value.
    """

    def __init__(self, labels: tuple[T, ...]):
        """
        Initializes the binary encoder with the given labels.

        Args:
            labels (tuple[T, ...]): A tuple of class labels.

        Raises: ValueError: If the number of labels is not equal to 2.
        """
        if len(labels) != 2:
            raise ValueError(f"Must be 2 labels to use binary encoding. Got {len(labels)}")

        super().__init__(labels)

    def encode(self, label: Tensor) -> Tensor[np.integer]:
        """
        Encodes a given label into a binary value.
        Args:
            label (Tensor): The label to encode.
        Returns:
            Tensor[np.integer]: The index of the label in the labels list.
        """
        xp = cp.get_array_module(label.data)
        out = xp.vectorize(self.labels.index)(label.data)
        return Tensor(out)

    def decode(self, encoded: Tensor[np.integer] | Tensor[np.floating]) -> T:
        """
        Decodes back to its corresponding class label.

        Args:
            encoded (Tensor[np.integer] | Tensor[np.floating]): The encoded value to decode. It can be a probability.

        Returns:
            T: The decoded class label corresponding to the encoded value.
        """
        return self.labels[round(encoded.item())]
