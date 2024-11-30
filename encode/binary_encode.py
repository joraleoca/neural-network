import numpy as np

from .encoder import Encoder
from core import Tensor


class BinaryEncoder(Encoder):
    """
    BinaryEncoder is a class that encodes categorical labels into a binary value.
    """

    classes: tuple[str, ...]

    def __post_init__(self):
        """
        Post-initialization method to validate the number of classes.
        Raises:
            ValueError: If the number of classes is not equal to 2.
        """
        if len(self.classes) != 2:
            raise ValueError(
                f"Must be 2 classes to use binary encoding. Got {len(self.classes)}"
            )

    def encode(self, label: str) -> int:
        """
        Encodes a given label into a binary value.
        Args:
            label (str): The label to encode.
        Returns:
            int: The index of the label in the classes list.
        """
        return self.classes.index(label)

    def decode(self, encoded: Tensor[np.integer] | Tensor[np.floating]) -> str:
        """
        Decodes back to its corresponding class label.

        Args:
            encoded (Tensor[int] | Tensor[float]): The encoded value to decode. It can be a probability.

        Returns:
            str: The decoded class label corresponding to the encoded value.
        """
        return self.classes[round(encoded.item())]
