import pytest

import numpy as np

from src.structure import Convolution
from src.activation import LeakyRelu, Sigmoid, Tanh
from src.initialization import LeCunNormal, HeNormal
from src.core import Tensor, op


class TestConvolution:
    convolution_type: type[Convolution] = Convolution

    def test_constructor(self):
        out_channels = 10
        kernel_shape = (1, 1)
        layer = self.convolution_type(out_channels, kernel_shape)

        assert layer.output_dim == out_channels, (
            f"Output channels should be 10. Got {layer.output_dim}."
        )
        assert layer.activation_function is None, (
            f"Activation function should be None. Got {layer.activation_function}."
        )
        assert layer.kernel_shape == kernel_shape, "Kernel"

    def test_constructor_channels(self):
        channels = (10, 100)
        layer = self.convolution_type(channels, (1, 1))

        assert layer.input_dim == channels[0], (
            f"Input channels should be 10. Got {layer.input_dim}."
        )
        assert layer.output_dim == channels[1], (
            f"Output channels should be 100. Got {layer.output_dim}."
        )
        assert layer.activation_function is None, (
            f"Activation function should be None. Got {layer.activation_function}."
        )

    @pytest.mark.parametrize(
        "channels",
        [
            (0, 10),
            (10, 0),
            (0, 0),
            (-1, 10),
            (10, -1),
            (-1, -1),
            0,
            -1,
            -100,
            (1, 2, 3),
        ],
        ids=[
            "zero_in_feature",
            "zero_out_feature",
            "zero_channels",
            "negative_in_feature",
            "negative_out_feature",
            "negative_channels",
            "zero",
            "-1_negative",
            "-100_negative",
            "invalid_shape",
        ],
    )
    def test_exception_invalid_channels(self, channels):
        with pytest.raises(ValueError):
            self.convolution_type(channels, (1, 1))

    @pytest.mark.parametrize(
        "activation_function",
        [
            # Choosed random activation functions
            LeakyRelu(),
            Sigmoid(),
            Tanh(),
        ],
    )
    def test_activation_function(self, activation_function):
        layer = self.convolution_type(
            10, (1, 1), activation_function=activation_function
        )

        assert layer.activation_function == activation_function, (
            f"Activation function should be {activation_function}. Got {layer.activation_function}."
        )

    @pytest.mark.parametrize(
        "initializer",
        [
            LeCunNormal(),
            HeNormal(),
        ],
        ids=lambda x: x.__class__.__name__,
    )
    def test_initializer(self, initializer):
        layer = self.convolution_type(10, (1, 1), initializer=initializer)

        assert layer.initializer == initializer, (
            f"Initializer should be {initializer}. Got {layer.initializer}."
        )

    def test_weights_initializated(self):
        channels = (10, 100)
        kernel_shape = (3, 3)
        layer = self.convolution_type((10, 100), (3, 3), initializer=LeCunNormal())

        assert hasattr(layer, "weights"), "Weights should be initialized."
        assert layer.weights.shape == (channels[1], 1, 1, channels[0], *kernel_shape), (
            f"Weights shape should be {(channels[1], 1, 1, channels[0], *kernel_shape)}. Got {layer.weights.shape}."
        )

    def test_induced_input_dim_and_weight_initializate(self):
        channels = 100
        kernel_shape = (3, 3)
        layer = self.convolution_type(channels, kernel_shape, initializer=LeCunNormal())

        data = op.zeros((3, 10, 200))
        layer.forward(data)

        assert layer.input_dim == 3, (
            f"Input channels should be 3. Got {layer.input_dim}."
        )
        assert hasattr(layer, "weights"), "Weights should be initialized."
        assert layer.weights.shape == (channels, 1, 1, data.shape[0]) + kernel_shape, (
            f"Weights shape should be {(channels, 1, 1, data.shape[0]) + kernel_shape}. Got {layer.weights.shape}."
        )

    def test_rng(self):
        channels = (10, 10)
        kernel_shape = (3, 3)
        layer1 = self.convolution_type(
            channels, kernel_shape, initializer=LeCunNormal(), rng=42
        )
        layer2 = self.convolution_type(
            channels, kernel_shape, initializer=LeCunNormal(), rng=42
        )

        assert layer1.rng == layer2.rng, "RNG should be the same for both layers."
        assert layer1.weights == layer2.weights, (
            "Weights should be the same for both layers."
        )

        layer3 = self.convolution_type(
            channels, kernel_shape, initializer=LeCunNormal(), rng=43
        )

        assert layer1.rng != layer3.rng, "RNG should be different for both layers."
        assert layer1.weights != layer3.weights, (
            "Weights should be different for both layers."
        )

    def test_forward(self):
        channels = (16, 8)
        kernel_shape = (3, 3)

        layer = self.convolution_type(
            channels, kernel_shape, initializer=LeCunNormal(), rng=0
        )

        data = Tensor(np.random.default_rng().random((channels[0], 100, 100)))
        output = layer.forward(data)

        assert isinstance(output, Tensor), "Output should be a Tensor."
        assert output.shape == (1, channels[-1], 98, 98), (
            f"Output shape should be (1, 8, 98, 98), but got {output.shape}."
        )

        assert hasattr(layer, "weights"), "Weights should be initialized."
        assert layer.weights.shape == (channels[1], 1, 1, channels[0], *kernel_shape), (
            f"Weights shape should be (8, 1, 1, 16, 3, 3), but got {layer.weights.shape}."
        )

        assert hasattr(layer, "biases"), "Biases should be initialized."
        assert layer.biases.shape == (channels[-1], 1, 1), (
            f"Biases shape should be (8, 1, 1), but got {layer.biases.shape}."
        )

    def test_forward_activation_function(self):
        channels = (16, 8)
        kernel_shape = (3, 3)

        layer = self.convolution_type(
            channels,
            kernel_shape,
            activation_function=LeakyRelu(),
            initializer=LeCunNormal(),
            rng=0,
        )

        layer2 = self.convolution_type(
            channels,
            kernel_shape,
            initializer=LeCunNormal(),
            rng=0,
        )

        data = Tensor(np.random.default_rng().random((channels[0], 100, 100)))
        output = layer.forward(data)
        output2 = layer2.forward(data)

        assert LeakyRelu()(output2) == output, (
            "Output should be the activation function of the forward output."
        )

    def test_forward_dimensions_mismatch(self):
        layer = self.convolution_type((10, 10), (3, 3), initializer=LeCunNormal())

        data = Tensor(np.random.default_rng().random((8, 10, 10)))

        with pytest.raises(ValueError):
            layer.forward(data)

    def test_forward_dimensions_mismatch_induced(self):
        layer = self.convolution_type(10, (3, 3), initializer=LeCunNormal())

        data = Tensor(np.random.default_rng().random((10, 10, 10)))

        layer.forward(data)

        with pytest.raises(ValueError):
            layer.forward(Tensor(np.random.default_rng().random((10, 9, 10, 10))))

    def test_backwards(self):
        channels = (16, 8)
        kernel_shape = (3, 3)

        layer = self.convolution_type(
            channels, kernel_shape, initializer=LeCunNormal(), rng=0
        )

        layer.requires_grad = True

        data = Tensor(np.random.default_rng().random((2, channels[0], 100, 100)))
        output = layer.forward(data)

        output.backward()

        assert layer.weights_grad.shape == layer.weights.shape
