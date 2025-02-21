from typing import ClassVar, Any

import numpy as np
from numpy.random import Generator

from .trainable import Trainable
from src.core import Tensor, op
from src.activation import ActivationFunction, activation_from_name
from src.initialization import Initializer, LeCunNormal
import src.constants as c


class Convolution(Trainable):
    """Convolution layer in a neural network."""

    __slots__ = [
        "_in_channels",
        "_out_channels",
        "kernel_size",
        "stride",
        "padding",
    ]

    _in_channels: int
    _out_channels: int

    kernel_size: tuple[int, int]

    stride: int
    padding: int

    required_fields: ClassVar[tuple[str]] = (c.WEIGHT_PREFIX, c.BIAS_PREFIX, c.ACTIVATION_PREFIX, "stride", "padding")

    def __init__(
        self,
        channels: int | tuple[int, int],
        kernel_size: tuple[int, int],
        activation_function: ActivationFunction | None = None,
        initializer: Initializer = LeCunNormal(),
        *,
        stride: int = 1,
        padding: int = 0,
        rng: Generator | None = None,
    ) -> None:
        """
        Initializes a new convolution layer in the neural network.

        Args:
            channels (int | tuple[int, int]):
                If int, the number of output channels\n
                If tuple, the number of (in channels, out channels)
            kernel_size (tuple[int, int]): The size of the kernel.
            initializer (Initializer): The initializer for the weights of this layer.
            stride (int): The stride of the convolution operation.
            padding (int): The padding of the convolution operation.
            rng (Generator | None): A random number generator instance for initializing weights.
        """
        if stride <= 0:
            raise ValueError(f"The stride value must be positive. Got {stride}")
        if padding < 0:
            raise ValueError(f"The padding value must be non-negative. Got {padding}")
        if len(kernel_size) != 2:
            raise ValueError(
                f"The kernel size must have 2 dimension. Got {len(kernel_size)}"
            )

        super().__init__()

        if isinstance(channels, tuple):
            self._in_channels, self._out_channels = channels

            if self._in_channels <= 0 or self._out_channels <= 0:
                raise ValueError(
                    f"The channels must be positive. Got in: {self._in_channels}, out: {self._out_channels}"
                )

            self.weights = initializer.initialize(
                (self._out_channels, self._in_channels) + kernel_size,
                rng=rng,
            )
        else:
            self._out_channels = channels

        if self._out_channels <= 0:
            raise ValueError(
                f"The out channels must be positive. Got {self._out_channels}"
            )

        self.activation_function = activation_function
        self.initializer = initializer
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.rng = rng

    def forward(
        self,
        data: Tensor[np.floating],
    ) -> Tensor[np.floating]:
        """
        Apply a convolution operation to the input data.
        Args:
            data (NDArray[np.floating]): A 3D numpy array with shape (channels, height, width) representing the input data.
        Returns:
            NDArray[np.floating]: A 3D numpy array with shape (_out_channels, output_height, output_width) representing the convolved output.
        Raises:
            ValueError:
                If the input data does not have 3 dimensions.
                If the number of input channels does not match the expected number of input channels.
        """
        if data.ndim != 4:
            if data.ndim == 3:
                data = op.expand_dims(data, 0)
            else:
                raise ValueError(
                    f"Expected 4D input (batch, channels, height, width). Got {data.shape}"
                )

        if hasattr(self, "_in_channels") and data.shape[1] != self._in_channels:
            raise ValueError(
                f"The input must have {self._in_channels} channels. Got {data.shape[1]}"
            )

        if not hasattr(self, "weights"):
            self._in_channels = data.shape[1]

            self.weights = self.initializer.initialize(
                (self._out_channels, self._in_channels) + self.kernel_size,
                requires_grad=self.requires_grad,
                rng=self.rng,
            )

        output_height, output_width = self._output_dimensions(data.shape[-2:])

        if not hasattr(self, "biases"):
            self.biases = op.zeros(
                (self._out_channels, output_height, output_width),
                requires_grad=self.requires_grad
            )

        kernel_height, kernel_width = self.weights.shape[-2:]
        batch_size = data.shape[0]

        pad_width = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
        data = op.pad(data, pad_width, value=data.dtype.type(0))

        windows = op.compose([
            data[
                :,
                :,
                i : i + kernel_height,
                j : j + kernel_width,
            ]
            for i in range(0, output_height * self.stride, self.stride)
            for j in range(0, output_width * self.stride, self.stride)
        ])

        windows = windows.reshape(
            (batch_size, output_height, output_width, self._in_channels, kernel_height, kernel_width)
        )

        out = [
            op.sum(self.weights[out_channel] * windows, axis=(-1, -2, -3)) + self.biases[out_channel]
            for out_channel in range(self._out_channels)
        ]
        
        out = op.compose(out)

        out = op.transpose(out, axes=(1, 0, 2, 3))

        if self.activation_function:
            out = self.activation_function(out)

        return out

    @property
    def input_dim(self) -> int:
        """Returns the number of input channels of the layer."""
        return self._in_channels

    @property
    def output_dim(self) -> int:
        """Returns the number of output channels of the layer."""
        return self._out_channels

    def _output_dimensions(self, input_size: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate the output dimensions of the convolution operation.
        Args:
            input_size (tuple[int, ...]): A tuple representing the height and width of the input.
        Returns:
            tuple[int, ...]: A tuple representing the height and width of the output.
        """
        output_size = tuple(
            ((d - self.weights.shape[-1 - i] + 2 * self.padding) // self.stride) + 1
            for i, d in enumerate(input_size)
        )

        return output_size

    def data_to_store(self) -> dict[str, Any]:
        return {
            c.WEIGHT_PREFIX: self.weights or None,
            c.BIAS_PREFIX: self.biases or None,
            c.ACTIVATION_PREFIX: self.activation_function.__class__.__name__ or None,
            "stride": self.stride or None,
            "padding": self.padding or None,
        }

    @staticmethod
    def from_data(data: dict[str, Any]) -> "Convolution":
        weights = data[c.WEIGHT_PREFIX]
        out_channels, in_channels, kernel_height, kernel_width = weights.shape

        layer = Convolution(
            (in_channels, out_channels), (kernel_height, kernel_width), stride=data["stride"], padding=data["padding"]
        )

        layer._in_channels = in_channels
        layer.weights = Tensor(weights)
        layer.biases = Tensor(data[c.BIAS_PREFIX])
        layer.activation_function = activation_from_name(data[c.ACTIVATION_PREFIX].item())()

        return layer
