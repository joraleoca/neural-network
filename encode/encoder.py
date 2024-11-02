from abc import ABC, abstractmethod
from typing import Any


class Encoder(ABC):
    """
    Encoder is an abstract base class that defines the structure for encoding labels into a specific format.
    Attributes:
        classes (tuple[str, ...]): A tuple containing the classes that can be encoded.
    """

    classes: tuple[str, ...]

    def __init__(self, classes: tuple[str, ...]) -> None:
        self.classes = classes

    def __call__(self, label: str) -> Any:
        return self.encode(label)

    @abstractmethod
    def encode(self, label: str) -> Any:
        """
        Encodes the given label into a specific format.

        Args:
            label (str): The label to be encoded.

        Returns:
            Any: The encoded representation of the label.
        """
        pass

    @abstractmethod
    def decode(self, encoded: Any) -> str:
        """
        Decodes the given encoded data into a string.

        Args:
            encoded (Any): The data to be decoded.

        Returns:
            str: The decoded string.
        """
        pass
