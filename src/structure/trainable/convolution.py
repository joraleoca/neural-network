from typing import ClassVar, Any

import numpy as np
from numpy.random import Generator

from .trainable import Trainable
from src.core import Tensor, op
from src.initialization import Initializer
import src.constants as c


class Convolution(Trainable):
    """Convolution layer in a neural network."""

    __slots__ = (
        "_in_channels",
        "_out_channels",
        "kernel_shape",
        "stride",
        "padding",
    )

    _in_channels: int
    _out_channels: int

    kernel_shape: tuple[int, int]

    stride: int
    padding: int

    required_fields: ClassVar[tuple[str, ...]] = (
        c.WEIGHT_PREFIX,
        c.BIAS_PREFIX,
        "stride",
        "padding",
    )

    def __init__(
        self,
        channels: int | tuple[int, int],
        kernel_shape: tuple[int, int],
        initializer: Initializer | None = None,
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
            kernel_shape (tuple[int, int]): The shape of the kernel.
            initializer (Initializer): The initializer for the weights of this layer.
            stride (int): The stride of the convolution operation.
            padding (int): The padding of the convolution operation.
            rng (Generator | None): A random number generator instance for initializing weights.
        """
        if stride <= 0:
            raise ValueError(f"The stride value must be positive. Got {stride}")
        if padding < 0:
            raise ValueError(f"The padding value must be non-negative. Got {padding}")
        if len(kernel_shape) != 2:
            raise ValueError(
                f"The kernel size must have 2 dimension. Got {len(kernel_shape)}"
            )

        super().__init__(initializer, rng=rng)
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.padding = padding

        if isinstance(channels, tuple):
            self._in_channels, self._out_channels = channels

            if self._in_channels <= 0 or self._out_channels <= 0:
                raise ValueError(
                    f"The channels must be positive. Got in: {self._in_channels}, out: {self._out_channels}"
                )

            if self._initializer is not None:
                self._initializate_weights()
        else:
            self._out_channels = channels

        if self._out_channels <= 0:
            raise ValueError(
                f"The out channels must be positive. Got {self._out_channels}"
            )

    def __call__(
        self,
        data: Tensor[np.floating],
    ) -> Tensor[np.floating]:
        """
        Apply a convolution operation to the input data.
        Args:
            data (Tensor[np.floating]): A 3D numpy array with shape (channels, height, width) representing the input data.
        Returns:
            Tensor[np.floating]: A 3D numpy array with shape (_out_channels, output_height, output_width) representing the convolved output.
        Raises:
            ValueError:
                If the input data does not have 3 dimensions.
                If the number of input channels does not match the expected number of input channels.
        """
        if data.ndim != 4:
            if data.ndim == 3:
                data = op.expand_dims(data, 0) #type: ignore
            else:
                raise ValueError(
                    f"Expected 4D input (batch, channels, height, width). Got {data.shape}"
                )
        
        if not hasattr(self, "weights"):
            self._in_channels = data.shape[1]
            self._initializate_weights() 

        if not hasattr(self, "biases"):
            self.biases = op.zeros(
                (self._out_channels, 1, 1),
                requires_grad=self.requires_grad,
            )

        if data.shape[1] != self._in_channels:
            raise ValueError(
                f"The input must have {self._in_channels} channels. Got {data.shape[1]}"
            )

        pad_width = (
            (0, 0),
            (0, 0),
            (self.padding, self.padding),
            (self.padding, self.padding),
        )
        data = op.pad(data, pad_width, value=data.dtype.type(0))

        windows = self._windows(data)

        out = op.sum(self.weights * windows, axis=(-1, -2, -3)) + self.biases

        if out.ndim == 3:
            out = op.expand_dims(out, 0)

        return out

    def _windows(self, data: Tensor) -> Tensor:
        batch_size, in_channels, in_height, in_width = data.shape
        kernel_height, kernel_width = self.kernel_shape

        out_height = 1 + ((in_height - kernel_height) // self.stride)
        out_width = 1 + ((in_width - kernel_width) // self.stride) 

        window_shape = (
            batch_size,
            1,
            out_height,
            out_width,
            in_channels,
            kernel_height,
            kernel_width,
        )   

        window_strides = (
            data.strides[0],
            0,
            data.strides[2] * self.stride,
            data.strides[3] * self.stride,
            data.strides[1],
            data.strides[2],
            data.strides[3],
        )
        
        return op.as_strided(
            data, 
            shape=window_shape, 
            strides=window_strides
        )

    def _initializate_weights(self) -> None:
        """Initializes the weights of the layer."""
        assert self.requires_grad is not None, (
            "Requires grad cannot be None when initializing weights."
        )

        assert self._initializer is not None, (
            "Initializer cannot be None when initializing weights."
        )

        self.weights = self._initializer.initialize(
            (self._out_channels, 1, 1, self._in_channels) + self.kernel_shape,
            requires_grad=self.requires_grad,
            rng=self.rng,
        )

        self._initializer = None


    @property
    def input_dim(self) -> int:
        """Returns the number of input channels of the layer."""
        return self._in_channels

    @property
    def output_dim(self) -> int:
        """Returns the number of output channels of the layer."""
        return self._out_channels

    def data_to_store(self) -> dict[str, Any]:
        return {
            c.WEIGHT_PREFIX: self.weights or None,
            c.BIAS_PREFIX: self.biases or None,
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

        return layer
