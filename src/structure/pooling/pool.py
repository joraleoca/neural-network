from abc import ABC
from typing import ClassVar, Any

from numpy import generic as np_generic
from numpy.typing import NDArray

from ..layer import Layer
from src.tensor import Tensor, op


class Pool(Layer, ABC):
    """Pool layer in a neural network."""

    __slots__ = [
        "channels",
        "filter_shape",
        "stride",
        "padding",
    ]

    channels: int

    filter_shape: tuple[int, int]
    stride: int
    padding: int

    required_fields: ClassVar[tuple[str, ...]] = ("channels", "filter_shape", "stride", "padding")

    def __init__(
        self,
        channels: int,
        filter_shape: tuple[int, int] | int,
        *,
        stride: int = 1,
        padding: int = 0,
    ):
        """
        Initializes a new layer in the neural network.
        Args:
            channels (int): The number of channels in the input and output.
            filter_shape (tuple[int, int] | int): The shape of the filter.
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
        
        if isinstance(filter_shape, int):
            filter_shape = (filter_shape, filter_shape)
        
        if any(i <= 0 for i in filter_shape):
            raise ValueError(f"The filter shape must be positive. Got {filter_shape}")


        self.channels = channels
        self.filter_shape = filter_shape
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

    def _pad[T: np_generic](self, data: Tensor[T], const_val: T) -> Tensor[T]:
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
    
    def _windows(self, data: Tensor) -> NDArray:
        """
        Extract the sliding windows from the input tensor.

        Args:
            data (NDArray): The input tensor from which the windows are extracted.
        
        Returns:
            NDArray: The sliding windows extracted from the input tensor.
        """
        batch_size, channels, in_height, in_width = data.shape
        filter_height, filter_width = self.filter_shape

        out_height = 1 + ((in_height - filter_height) // self.stride)
        out_width = 1 + ((in_width - filter_width) // self.stride)

        shape = (batch_size, channels, out_height, out_width, filter_height, filter_width)

        strides = (
            data.strides[0],
            data.strides[1],
            data.strides[2] * self.stride,
            data.strides[3] * self.stride,
            data.strides[2],
            data.strides[3],
        )

        return op.as_strided(data, shape=shape, strides=strides)

    def _output_dimensions(self, input_size: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate the output dimensions of the pooling operation.
        Args:
            input_size (tuple[int, ...]): A tuple representing the height and width of the input.
        Returns:
            tuple[int, ...]: A tuple representing the height and width of the output.
        """
        output_size: tuple[int, ...] = tuple(
            ((d - self.filter_shape[-1 - i] + 2 * self.padding) // self.stride) + 1
            for i, d in enumerate(input_size)
        )

        return output_size

    def data_to_store(self) -> dict[str, Any]:
        return {
            "channels": self.channels,
            "filter_shape": self.filter_shape,
            "stride": self.stride,
            "padding": self.padding,
        }
