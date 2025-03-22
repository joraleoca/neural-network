from typing import Any

import numpy as np

from .trainable import Trainable
from src.tensor import Tensor, op
from src.initialization import Initializer, HeUniform


class Convolution(Trainable):
    """Convolution layer in a neural network."""

    __slots__ = (
        "in_channels",
        "out_channels",
        "kernel_shape",
        "stride",
        "padding",
    )

    in_channels: int
    out_channels: int

    kernel_shape: tuple[int, int]

    stride: int
    padding: int

    def __init__(
        self,
        channels: int | tuple[int, int],
        kernel_shape: tuple[int, int],
        initializer: Initializer = HeUniform(),
        *,
        stride: int = 1,
        padding: int = 0,
        rng: Any = None,
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
            rng (Any): A random number generator instance for initializing weights.
        """
        if stride <= 0:
            raise ValueError(f"The stride value must be positive. Got {stride}")
        if padding < 0:
            raise ValueError(f"The padding value must be non-negative. Got {padding}")
        if len(kernel_shape) != 2:
            raise ValueError(f"The kernel size must have 2 dimension. Got {len(kernel_shape)}")

        super().__init__(initializer, rng=rng)
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.padding = padding

        if isinstance(channels, tuple):
            self.in_channels, self.out_channels = channels

            if self.in_channels <= 0 or self.out_channels <= 0:
                raise ValueError(f"The channels must be positive. Got in: {self.in_channels}, out: {self.out_channels}")

            self._initializate_weights()
        else:
            self.out_channels = channels

            if self.out_channels <= 0:
                raise ValueError(f"The out channels must be positive. Got {self.out_channels}")

        self.biases.set_data(
            op.zeros(
                (self.out_channels, 1, 1),
                requires_grad=self.requires_grad,
            )
        )

    def __call__(
        self,
        data: Tensor[np.floating],
    ) -> Tensor[np.floating]:
        """
        Apply a convolution operation to the input data.
        Args:
            data (Tensor[np.floating]): A 4D numpy array with shape (batch size, channels, height, width) representing the input data.
        Returns:
            Tensor[np.floating]: A 4D numpy array with shape (batch size, out_channels, output_height, output_width) representing the convolved output.
        Raises:
            ValueError:
                If the input data does not have 4 dimensions.
                If the number of input channels does not match the expected number of input channels.
        """
        if data.ndim != 4:
            raise ValueError(f"Expected 4D input (batch, channels, height, width). Got {data.shape}")

        if self._initializer is not None:
            self.in_channels = data.shape[1]
            self._initializate_weights()

        if data.shape[1] != self.in_channels:
            raise ValueError(f"The input must have {self.in_channels} channels. Got {data.shape[1]}")

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

        return op.as_strided(data, shape=window_shape, strides=window_strides)

    def _initializate_weights(self) -> None:
        """Initializes the weights of the layer."""
        assert self._initializer is not None, "Initializer cannot be None when initializing weights."

        self.weights.set_data(
            self._initializer.initialize(
                (self.out_channels, 1, 1, self.in_channels) + self.kernel_shape,
                requires_grad=self.requires_grad,
                rng=self.rng,
            )
        )

        self._initializer = None
