import pytest
from structure import Layer
from activation import Sigmoid, Softmax, Relu, Tanh
from initialization import HeUniform, XavierNormal
from encode import Encoder
from config.ff_network import FeedForwardConfig


def test_feedforward_config_with_list_of_ints():
    config = FeedForwardConfig(
        network_structure=[10, 20, 30],
        classes=("class1", "class2"),
        hidden_activation=Sigmoid(),
        output_activation=Sigmoid(),
        initializer=HeUniform(),
        random_seed=42,
    )

    assert len(config.network_structure) == 3
    assert all(isinstance(layer, Layer) for layer in config.network_structure)
    assert config.network_structure[0].out_features == 10
    assert config.network_structure[1].out_features == 20
    assert config.network_structure[2].out_features == 30
    assert isinstance(config.network_structure[0].activation_function, Sigmoid)
    assert isinstance(config.network_structure[1].activation_function, Sigmoid)
    assert isinstance(config.network_structure[2].activation_function, Sigmoid)


def test_feedforward_config_with_list_of_layers():
    layers = [Layer(features=10), Layer(features=20), Layer(features=30)]
    config = FeedForwardConfig(
        network_structure=layers,
        classes=("class1", "class2"),
        hidden_activation=Sigmoid(),
        output_activation=Sigmoid(),
        initializer=HeUniform(),
        random_seed=42,
    )

    assert len(config.network_structure) == 3
    assert config.network_structure == layers


def test_feedforward_config_with_list_of_tuples():
    config = FeedForwardConfig(
        network_structure=[Layer((10, 20)), Layer((20, 30)), Layer((30, 40))],
        classes=("class1", "class2"),
        hidden_activation=Sigmoid(),
        output_activation=Sigmoid(),
        initializer=HeUniform(),
        random_seed=42,
    )

    assert len(config.network_structure) == 3
    assert all(isinstance(layer, Layer) for layer in config.network_structure)
    assert config.network_structure[0].in_features == 10
    assert config.network_structure[0].out_features == 20
    assert config.network_structure[1].in_features == 20
    assert config.network_structure[1].out_features == 30
    assert config.network_structure[2].in_features == 30
    assert config.network_structure[2].out_features == 40
    assert isinstance(config.network_structure[0].activation_function, Sigmoid)
    assert isinstance(config.network_structure[1].activation_function, Sigmoid)
    assert isinstance(config.network_structure[2].activation_function, Sigmoid)


def test_feedforward_config_with_list_of_layers_and_activations():
    layers = [
        Layer(features=10, activation_function=Relu()),
        Layer(features=20, activation_function=None),
        Layer(features=20, activation_function=Softmax()),
        Layer(features=30, activation_function=None),
    ]
    config = FeedForwardConfig(
        network_structure=layers,
        classes=("class1", "class2"),
        hidden_activation=Tanh(),
        output_activation=Sigmoid(),
        initializer=HeUniform(),
        random_seed=42,
    )

    assert len(config.network_structure) == 4
    assert config.network_structure == layers

    assert isinstance(config.network_structure[0].activation_function, Relu)
    assert isinstance(config.network_structure[1].activation_function, Tanh)
    assert isinstance(config.network_structure[2].activation_function, Softmax)
    assert isinstance(config.network_structure[3].activation_function, Sigmoid)


def test_feedforward_config_with_incomplete_list_of_layers():
    with pytest.raises(ValueError):
        FeedForwardConfig(
            network_structure=[
                Layer(features=10),
                Layer(features=0),  # Invalid layer with 0 features
                Layer(features=30),
            ],
            classes=("class1", "class2"),
            hidden_activation=Sigmoid(),
            output_activation=Sigmoid(),
            initializer=HeUniform(),
            random_seed=42,
        )


def test_feedforward_config_missing_required_attributes():
    with pytest.raises(TypeError):
        FeedForwardConfig(
            network_structure=[10, 20, 30],
            hidden_activation=Sigmoid(),
            output_activation=Sigmoid(),
            initializer=HeUniform(),
            random_seed=42,
        )


def test_feedforward_config_invalid_types():
    with pytest.raises(TypeError):
        FeedForwardConfig(
            network_structure="invalid_type",
            classes=("class1", "class2"),
            hidden_activation=Sigmoid(),
            output_activation=Sigmoid(),
            initializer=HeUniform(),
            random_seed=42,
        )


def test_feedforward_config_with_valid_encoder():
    class CustomEncoder(Encoder):
        pass

    config = FeedForwardConfig(
        network_structure=[10, 20, 30],
        classes=("class1", "class2"),
        hidden_activation=Sigmoid(),
        output_activation=Sigmoid(),
        initializer=HeUniform(),
        encoder=CustomEncoder,
        random_seed=42,
    )

    assert config.encoder == CustomEncoder


def test_feedforward_config_with_different_initializers():
    config_he = FeedForwardConfig(
        network_structure=[10, 20, 30],
        classes=("class1", "class2"),
        hidden_activation=Sigmoid(),
        output_activation=Sigmoid(),
        initializer=HeUniform(),
        random_seed=42,
    )

    config_xavier = FeedForwardConfig(
        network_structure=[10, 20, 30],
        classes=("class1", "class2"),
        hidden_activation=Sigmoid(),
        output_activation=Sigmoid(),
        initializer=XavierNormal(),
        random_seed=42,
    )

    assert isinstance(config_he.initializer, HeUniform)
    assert isinstance(config_xavier.initializer, XavierNormal)


def test_feedforward_config_with_layer_initializer():
    config = FeedForwardConfig(
        network_structure=[
            Layer(features=10, weights_initializer=HeUniform()),
            Layer(features=20, weights_initializer=XavierNormal()),
            Layer(features=30),
        ],
        classes=("class1", "class2"),
        hidden_activation=Sigmoid(),
        output_activation=Sigmoid(),
        random_seed=42,
    )

    assert isinstance(config.network_structure[0]._initializer, HeUniform)
    assert isinstance(config.network_structure[1]._initializer, XavierNormal)
    assert isinstance(config.network_structure[2]._initializer, HeUniform)
