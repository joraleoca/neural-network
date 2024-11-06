from abc import ABC, abstractmethod
from typing import Any


class Encoder[T](ABC):
    """
    Encoder is an abstract base class that defines the structure for encoding labels into a specific format.
    Attributes:
        classes (tuple[str, ...]): A tuple containing the classes that can be encoded.
    """

    classes: tuple[str, ...]

    def __init__(self, classes: tuple[str, ...]) -> None:
        """
        Initializes the encoder with the given classes.

        Args:
            classes (tuple[str, ...]): A tuple containing the class labels.
        """
        self.classes = classes

    def __call__(self, label: str) -> T:
        """
        Calls the encode method on the given label.

        Args:
            label (str): The label to be encoded.

        Returns:
            T: The encoded label.
        """
        return self.encode(label)

    @abstractmethod
    def encode(self, label: str) -> T:
        """
        Encodes the given label into a specific format.
        Args:
            label (str): The label to be encoded.
        Returns:
            T: The encoded representation of the label.
        """
        pass

    @abstractmethod
    def decode(self, encoded: T | Any) -> str:
        """
        Decodes the given encoded data into a string.

        The `encoded` parameter can be of type `T` or any type that is "T-like" representing a probability.

        Args:
            encoded (T | Any): The data to be decoded. Can be of type `T` or a "T-like" probability.

        Returns:
            str: The decoded string.
        """
        pass
