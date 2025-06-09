import pytest

import numpy as np

from src.structure import Convolution
from src.initialization import LeCunNormal, HeNormal
from src.tensor import Tensor, op


def test_constructor():
    out_channels = 10
    kernel_shape = (1, 1)
    layer = Convolution(out_channels, kernel_shape)

    assert layer.out_channels == out_channels, f"Layer must have {out_channels} out channels. Got {layer.out_channels}"
    assert layer.kernel_shape == kernel_shape, f"Kernel shape must be {kernel_shape}. Got {layer.kernel_shape}"


def test_constructor_channels():
    channels = (10, 100)
    layer = Convolution(channels, (1, 1))

    assert layer.in_channels == channels[0], f"Input channels should be {channels[0]}. Got {layer.in_channels}."
    assert layer.out_channels == channels[1], f"Output channels should be {channels[1]}. Got {layer.out_channels}."


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
def test_exception_invalid_channels(channels):
    with pytest.raises(ValueError):
        Convolution(channels, (1, 1))


@pytest.mark.parametrize(
    "initializer",
    [
        LeCunNormal(),
        HeNormal(),
    ],
    ids=lambda x: x.__class__.__name__,
)
def test_initializer(initializer):
    layer = Convolution(10, (1, 1), initializer=initializer)

    assert layer.initializer == initializer, f"Initializer should be {initializer}. Got {layer.initializer}."


def test_weights_initializated():
    channels = (10, 100)
    kernel_shape = (3, 3)
    layer = Convolution((10, 100), (3, 3), initializer=LeCunNormal())

    assert hasattr(layer, "weights"), "Weights should be initialized."
    assert layer.weights.shape == (channels[1], 1, 1, channels[0], *kernel_shape), (
        f"Weights shape should be {(channels[1], 1, 1, channels[0], *kernel_shape)}. Got {layer.weights.shape}."
    )


def test_induced_input_dim_and_weight_initializate():
    channels = 100
    kernel_shape = (3, 3)
    layer = Convolution(channels, kernel_shape, initializer=LeCunNormal())

    data = op.zeros((1, 3, 10, 200))
    layer(data)

    assert layer.in_channels == data.shape[1], f"Input channels should be {data.shape[1]}. Got {layer.in_channels}."
    assert hasattr(layer, "weights"), "Weights should be initialized."
    assert layer.weights.shape == (channels, 1, 1, 3) + kernel_shape, (
        f"Weights shape should be {(channels, 1, 1, 3) + kernel_shape}. Got {layer.weights.shape}."
    )


def test_rng():
    channels = (10, 10)
    kernel_shape = (3, 3)
    layer1 = Convolution(channels, kernel_shape, initializer=LeCunNormal(), rng=42)
    layer2 = Convolution(channels, kernel_shape, initializer=LeCunNormal(), rng=42)

    assert layer1.rng == layer2.rng, "RNG should be the same for both layers."
    assert layer1.weights == layer2.weights, "Weights should be the same for both layers."

    layer3 = Convolution(channels, kernel_shape, initializer=LeCunNormal(), rng=43)

    assert layer1.rng != layer3.rng, "RNG should be different for both layers."
    assert layer1.weights != layer3.weights.data, "Weights should be different for both layers."


def test_forward_generic():
    channels = (16, 8)
    kernel_shape = (3, 3)

    layer = Convolution(channels, kernel_shape, initializer=LeCunNormal(), rng=0)

    data = Tensor(np.random.default_rng().random((1, channels[0], 100, 100)))
    output = layer(data)

    assert isinstance(output, Tensor), "Output should be a Tensor."
    assert output.shape == (1, channels[-1], 98, 98), f"Output shape should be (1, 8, 98, 98), but got {output.shape}."

    assert hasattr(layer, "weights"), "Weights should be initialized."
    assert layer.weights.shape == (channels[1], 1, 1, channels[0], *kernel_shape), (
        f"Weights shape should be (8, 1, 1, 16, 3, 3), but got {layer.weights.shape}."
    )

    assert hasattr(layer, "biases"), "Biases should be initialized."
    assert layer.biases.shape == (channels[-1], 1, 1), (
        f"Biases shape should be (8, 1, 1), but got {layer.biases.shape}."
    )


def test_forward_activation_function():
    channels = (16, 8)
    kernel_shape = (3, 3)

    layer = Convolution(
        channels,
        kernel_shape,
        initializer=LeCunNormal(),
        rng=0,
    )

    layer2 = Convolution(
        channels,
        kernel_shape,
        initializer=LeCunNormal(),
        rng=0,
    )

    data = Tensor(np.random.default_rng().random((1, channels[0], 100, 100)))
    output = layer(data)
    output2 = layer2(data)

    assert output2 == output, "Output should be the activation function of the forward output."


def test_forward_dimensions_mismatch():
    layer = Convolution((10, 10), (3, 3), initializer=LeCunNormal())

    data = Tensor(np.random.default_rng().random((8, 10, 10)))

    with pytest.raises(ValueError):
        layer(data)


def test_forward_dimensions_mismatch_induced():
    layer = Convolution(10, (3, 3), initializer=LeCunNormal())

    data = Tensor(np.random.default_rng().random((1, 10, 10, 10)))

    layer(data)

    with pytest.raises(ValueError):
        layer(Tensor(np.random.default_rng().random((10, 9, 10, 10))))


def test_backwards():
    channels = (16, 8)
    kernel_shape = (3, 3)

    layer = Convolution(channels, kernel_shape, initializer=LeCunNormal(), rng=0)

    layer.requires_grad = True

    data = Tensor(np.random.default_rng().random((2, channels[0], 100, 100)))
    output = layer(data)

    output.backward()

    assert layer.weights.grad is not None, "Weights should have gradients."
    assert layer.biases.grad is not None, "Biases should have gradients."
    assert layer.weights.grad.shape == layer.weights.shape, (
        f"Weights gradients shape should be {layer.weights.shape}. Got {layer.weights.grad.shape}."
    )
    assert layer.biases.grad.shape == layer.biases.shape, (
        f"Biases gradients shape should be {layer.biases.shape}. Got {layer.biases.grad.shape}."
    )
