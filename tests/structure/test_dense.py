import pytest

from src.structure import Dense
from src.initialization import LeCunNormal, HeNormal
from src.tensor import Tensor, op


def test_constructor_out_features():
    layer = Dense(10)

    assert layer.out_features == 10, f"Output features should be 10. Got {layer.out_features}."


def test_constructor_features():
    features = (10, 100)
    layer = Dense(features)

    assert layer.in_features == 10, f"Input features should be {features[0]}. Got {layer.in_features}."
    assert layer.out_features == 100, f"Output features should be {features[1]}. Got {layer.out_features}."


@pytest.mark.parametrize(
    "features",
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
        "zero_features",
        "negative_in_feature",
        "negative_out_feature",
        "negative_features",
        "zero",
        "-1_negative",
        "-100_negative",
        "invalid_features",
    ],
)
def test_exception_invalid_features(features):
    with pytest.raises(ValueError):
        Dense(features)


@pytest.mark.parametrize(
    "initializer",
    [
        LeCunNormal(),
        HeNormal(),
    ],
    ids=lambda x: x.__class__.__name__,
)
def test_initializer(initializer):
    layer = Dense(10, initializer=initializer)

    assert layer.initializer == initializer, f"Initializer should be {initializer}. Got {layer.initializer}."


def test_weights_initializated():
    features = (10, 100)
    layer = Dense(features, initializer=LeCunNormal())

    assert hasattr(layer, "weights"), "Weights should be initialized."
    assert layer.weights.shape == features, f"Weights shape should be {features}. Got {layer.weights.shape}."


def test_induced_input_dim_and_weight_initializate():
    layer = Dense(10, initializer=LeCunNormal())

    layer(Tensor([[1, 2, 3]]))

    assert layer.in_features == 3, f"Input features should be 3. Got {layer.in_features}."
    assert hasattr(layer, "weights"), "Weights should be initialized."
    assert layer.weights.shape == (3, 10), f"Weights shape should be (3, 10). Got {layer.weights.shape}."


def test_rng():
    layer1 = Dense((10, 10), initializer=LeCunNormal(), rng=42)
    layer2 = Dense((10, 10), initializer=LeCunNormal(), rng=42)

    assert layer1.rng == layer2.rng, "RNG should be the same for both layers."
    assert layer1.weights == layer2.weights, "Weights should be the same for both layers."

    layer3 = Dense((10, 10), initializer=LeCunNormal(), rng=43)

    assert layer1.rng != layer3.rng, "RNG should be different for both layers."
    assert layer1.weights != layer3.weights, "Weights should be different for both layers."


def test_forward():
    layer = Dense(10, initializer=LeCunNormal())

    data = Tensor([[1, 2, 3]])
    output = layer(data)

    assert isinstance(output, Tensor), "Output should be a Tensor."
    assert output.shape == (1, 10), f"Output shape should be (1, 10). Got {output.shape}."
    assert hasattr(layer, "weights"), "Weights should be initialized."
    assert layer.weights.shape == (3, 10), f"Weights shape should be (3, 10). Got {layer.weights.shape}."
    assert hasattr(layer, "biases"), "Biases should be initialized."
    assert layer.biases.shape == (1, 10), f"Biases shape should be (1, 10). Got {layer.biases.shape}."
    assert output == (data @ layer.weights) + layer.biases, (
        "Output data should be the dot product of data and weights plus biases."
    )


def test_forward_dimensions_mismatch():
    layer = Dense((10, 10), initializer=LeCunNormal())

    data = Tensor([[1, 2, 3, 4]])

    with pytest.raises(ValueError):
        layer(data)


def test_forward_dimensions_mismatch_induced():
    layer = Dense(10, initializer=LeCunNormal())

    data = Tensor([[1, 2, 3, 4]])

    layer(data)

    with pytest.raises(ValueError):
        layer(Tensor([[1, 2, 3, 4, 5, 6]]))


def test_backwards():
    layer = Dense(10, initializer=LeCunNormal())

    layer.requires_grad = True

    data = Tensor([[1, 2, 3]])
    output = layer(data)

    output.backward()

    grad = op.zeros_like(output)
    grad.fill(1)
    assert data.T * grad == layer.weights.grad, "Grad input should be the dot product of grad and weights transposed."
