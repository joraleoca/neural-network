from abc import ABC
from typing import ClassVar, Any

from ..layer import Layer
from src.core import Tensor, op


class Pool(Layer, ABC):
    """Pool layer in a neural network."""

    __slots__ = [
        "channels",
        "filter_size",
        "stride",
        "padding",
    ]

    channels: int

    filter_size: tuple[int, int]
    stride: int
    padding: int

    required_fields: ClassVar[set[str]] = ("channels", "filter_size", "stride", "padding")

    def __init__(
        self,
        channels: int,
        filter_size: tuple[int, int],
        *,
        stride: int = 1,
        padding: int = 0,
    ):
        """
        Initializes a new layer in the neural network.
        Args:
            channels (int): The number of channels in the input and output.
            filter_size (tuple[int, int]): The size of the filter.
            stride (int): The stride of the sliding window.
            padding (int): The padding of the sliding window.
        Raises:
            ValueError: If any parameter is incorrect.
        """
        if channels <= 0:
            raise ValueError(f"The pooling must have positive channels. Got {channels}")
        if stride <= 0:
            raise ValueError(f"The stride value must be positive. Got {stride}")
        if padding < 0:
            raise ValueError(f"The padding value must be non-negative. Got {padding}")

        self.channels = channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

    @property
    def input_dim(self) -> int:
        """Returns the number of input channels of the layer."""
        return self.channels

    @property
    def output_dim(self) -> int:
        """Returns the number of output channels of the layer."""
        return self.channels

    def _pad[T](self, data: Tensor[T], const_val: T) -> Tensor[T]:
        """
        Pad the input tensor with the specified padding value.

        Args:
            data (Tensor):
                The input tensor to be padded.
        Returns:
            Tensor: The padded tensor.
        Details:
            This method pads the input tensor `data` with the padding value specified in the instance variable `self.padding`.
            The padding is applied symmetrically along the height and width dimensions.
        """
        pad_width = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
        return op.pad(data, pad_width, value=const_val)

    def _output_dimensions(self, input_size: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate the output dimensions of the pooling operation.
        Args:
            input_size (tuple[int, ...]): A tuple representing the height and width of the input.
        Returns:
            tuple[int, ...]: A tuple representing the height and width of the output.
        """
        output_size: tuple[int, ...] = tuple(
            ((d - self.filter_size[-1 - i] + 2 * self.padding) // self.stride) + 1
            for i, d in enumerate(input_size)
        )

        return output_size

    def data_to_store(self) -> dict[str, Any]:
        return {
            "channels": self.channels,
            "filter_size": self.filter_size,
            "stride": self.stride,
            "padding": self.padding,
        }
