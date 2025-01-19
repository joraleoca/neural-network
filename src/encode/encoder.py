from abc import ABC, abstractmethod
from typing import Any

class Encoder[T, encode_type](ABC):
    """
    Encoder is an abstract base class that defines the structure for encoding labels into a specific format.
    """
    __slots__ = ["classes"]

    classes: tuple[T, ...]

    def __init__(self, classes: tuple[T, ...]) -> None:
        """
        Initializes the encoder with the given classes.
        Args:
            classes (tuple[T, ...]): A tuple containing the class labels.
        """
        self.classes = classes

    def __call__(self, label: T) -> encode_type:
        """
        Calls the encode method on the given label.
        Args:
            label (T): The label to be encoded.
        Returns:
            T: The encoded label.
        """
        return self.encode(label)

    @abstractmethod
    def encode(self, label: T) -> encode_type:
        """
        Encodes the given label into a specific format.
        Args:
            label (T): The label to be encoded.
        Returns:
            T: The encoded representation of the label.
        """
        pass

    @abstractmethod
    def decode(self, encoded: encode_type | Any) -> T:
        """
        Decodes the given encoded data into a string.
        Args:
            encoded (T | Any): The data to be decoded. Can be of type `encode_type` or "encode_type-like" probability.
        Returns:
            str: The decoded string.
        """
        pass
