import pytest

from src.structure import Dense, Convolution, MaxPool, Trainable
from src.activation import Sigmoid, Softmax, Relu, Tanh
from src.initialization import HeUniform, XavierNormal
from src.encode import Encoder
from src.config.ff_network import FeedForwardConfig


class TestDense:
    def test_feedforward_config_with_list_of_ints(self):
        config = FeedForwardConfig(
            network_structure=[10, 20, 30],
            classes=("class1", "class2"),
            hidden_activation=Sigmoid(),
            output_activation=Sigmoid(),
            initializer=HeUniform(),
            random_seed=42,
        )

        layers: list[Trainable] = [layer for layer in config.network_structure]

        assert len(config.network_structure) == 3
        assert all(isinstance(layer, Dense) for layer in config.network_structure)
        assert layers[0].output_dim == 10
        assert layers[1].output_dim == 20
        assert layers[2].output_dim == 30
        assert isinstance(layers[0].activation_function, Sigmoid)
        assert isinstance(layers[1].activation_function, Sigmoid)
        assert isinstance(layers[2].activation_function, Sigmoid)

    def test_feedforward_config_with_list_of_layers(self):
        layers = [
            Dense(features=10),
            Dense(features=20),
            Dense(features=30),
        ]
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

    def test_feedforward_config_with_list_of_tuples(self):
        config = FeedForwardConfig(
            network_structure=[
                Dense((10, 20)),
                Dense((20, 30)),
                Dense((30, 40)),
            ],
            classes=("class1", "class2"),
            hidden_activation=Sigmoid(),
            output_activation=Sigmoid(),
            initializer=HeUniform(),
            random_seed=42,
        )

        layers: list[Trainable] = [layer for layer in config.network_structure]

        assert len(config.network_structure) == 3
        assert all(isinstance(layer, Dense) for layer in config.network_structure)
        assert layers[0].input_dim == 10
        assert layers[0].output_dim == 20
        assert layers[1].input_dim == 20
        assert layers[1].output_dim == 30
        assert layers[2].input_dim == 30
        assert layers[2].output_dim == 40
        assert isinstance(layers[0].activation_function, Sigmoid)
        assert isinstance(layers[1].activation_function, Sigmoid)
        assert isinstance(layers[2].activation_function, Sigmoid)

    def test_feedforward_config_with_list_of_layers_and_activations(self):
        layers = [
            Dense(features=10, activation_function=Relu()),
            Dense(features=20, activation_function=None),
            Dense(features=20, activation_function=Softmax()),
            Dense(features=30, activation_function=None),
        ]
        config = FeedForwardConfig(
            network_structure=layers,
            classes=("class1", "class2"),
            hidden_activation=Tanh(),
            output_activation=Sigmoid(),
            initializer=HeUniform(),
            random_seed=42,
        )

        layers: list[Trainable] = [layer for layer in config.network_structure]

        assert len(config.network_structure) == 4
        assert config.network_structure == layers

        assert isinstance(layers[0].activation_function, Relu)
        assert isinstance(layers[1].activation_function, Tanh)
        assert isinstance(layers[2].activation_function, Softmax)
        assert isinstance(layers[3].activation_function, Sigmoid)

    def test_feedforward_config_with_incomplete_list_of_layers(self):
        with pytest.raises(ValueError):
            FeedForwardConfig(
                network_structure=[
                    Dense(features=10),
                    Dense(features=0),  # Invalid layer with 0 features
                    Dense(features=30),
                ],
                classes=("class1", "class2"),
                hidden_activation=Sigmoid(),
                output_activation=Sigmoid(),
                initializer=HeUniform(),
                random_seed=42,
            )

    def test_feedforward_config_missing_required_attributes(self):
        with pytest.raises(TypeError):
            FeedForwardConfig(
                network_structure=[10, 20, 30],
                hidden_activation=Sigmoid(),
                output_activation=Sigmoid(),
                initializer=HeUniform(),
                random_seed=42,
            )

    def test_feedforward_config_invalid_types(self):
        with pytest.raises(TypeError):
            FeedForwardConfig(
                network_structure="invalid_type",
                classes=("class1", "class2"),
                hidden_activation=Sigmoid(),
                output_activation=Sigmoid(),
                initializer=HeUniform(),
                random_seed=42,
            )

    def test_feedforward_config_with_valid_encoder(self):
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

    def test_feedforward_config_with_different_initializers(self):
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

    def test_feedforward_config_with_layer_initializer(self):
        config = FeedForwardConfig(
            network_structure=[
                Dense(features=10, weights_initializer=HeUniform()),
                Dense(features=20, weights_initializer=XavierNormal()),
                Dense(features=30),
            ],
            classes=("class1", "class2"),
            hidden_activation=Sigmoid(),
            output_activation=Sigmoid(),
            random_seed=42,
        )

        layers: list[Trainable] = [layer for layer in config.network_structure]

        assert isinstance(layers[0]._initializer, HeUniform)
        assert isinstance(layers[1]._initializer, XavierNormal)
        assert isinstance(layers[2]._initializer, HeUniform)


class TestConvultion:
    def test_feedforward_config_with_list_of_ints(self):
        config = FeedForwardConfig(
            network_structure=[10, 20, 30],
            classes=("class1", "class2"),
            hidden_activation=Sigmoid(),
            output_activation=Sigmoid(),
            initializer=HeUniform(),
            random_seed=42,
        )

        layers: list[Trainable] = [layer for layer in config.network_structure]

        assert len(config.network_structure) == 3
        assert all(isinstance(layer, Dense) for layer in config.network_structure)
        assert layers[0].output_dim == 10
        assert layers[1].output_dim == 20
        assert layers[2].output_dim == 30
        assert isinstance(layers[0].activation_function, Sigmoid)
        assert isinstance(layers[1].activation_function, Sigmoid)
        assert isinstance(layers[2].activation_function, Sigmoid)

    def test_feedforward_config_with_list_of_layers(self):
        layers = [
            Convolution(channels=10, kernel_size=(3, 3)),
            Convolution(channels=20, kernel_size=(3, 3)),
            Convolution(channels=30, kernel_size=(3, 3)),
        ]
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

    def test_feedforward_config_with_list_of_tuples(self):
        config = FeedForwardConfig(
            network_structure=[
                Convolution((10, 20), kernel_size=(3, 3)),
                Convolution((20, 30), kernel_size=(3, 3)),
                Convolution((30, 40), kernel_size=(3, 3)),
            ],
            classes=("class1", "class2"),
            hidden_activation=Sigmoid(),
            output_activation=Sigmoid(),
            initializer=HeUniform(),
            random_seed=42,
        )

        layers: list[Trainable] = [layer for layer in config.network_structure]

        assert len(config.network_structure) == 3
        assert all(isinstance(layer, Convolution) for layer in config.network_structure)
        assert layers[0].input_dim == 10
        assert layers[0].output_dim == 20
        assert layers[1].input_dim == 20
        assert layers[1].output_dim == 30
        assert layers[2].input_dim == 30
        assert layers[2].output_dim == 40
        assert isinstance(layers[0].activation_function, Sigmoid)
        assert isinstance(layers[1].activation_function, Sigmoid)
        assert isinstance(layers[2].activation_function, Sigmoid)

    def test_feedforward_config_with_list_of_layers_and_activations(self):
        layers = [
            Convolution(10, kernel_size=(3, 3), activation_function=Relu()),
            Convolution(20, kernel_size=(3, 3), activation_function=None),
            Convolution(20, kernel_size=(3, 3), activation_function=Softmax()),
            Convolution(30, kernel_size=(3, 3), activation_function=None),
        ]
        config = FeedForwardConfig(
            network_structure=layers,
            classes=("class1", "class2"),
            hidden_activation=Tanh(),
            output_activation=Sigmoid(),
            initializer=HeUniform(),
            random_seed=42,
        )

        layers: list[Trainable] = [layer for layer in config.network_structure]

        assert len(config.network_structure) == 4
        assert config.network_structure == layers

        assert isinstance(layers[0].activation_function, Relu)
        assert isinstance(layers[1].activation_function, Tanh)
        assert isinstance(layers[2].activation_function, Softmax)
        assert isinstance(layers[3].activation_function, Sigmoid)

    def test_feedforward_config_with_incomplete_list_of_layers(self):
        with pytest.raises(ValueError):
            FeedForwardConfig(
                network_structure=[
                    Convolution(10, (3, 3)),
                    Convolution(0, (3, 3)),  # Invalid layer with 0 channels
                    Convolution(30, (3, 3)),
                ],
                classes=("class1", "class2"),
                hidden_activation=Sigmoid(),
                output_activation=Sigmoid(),
                initializer=HeUniform(),
                random_seed=42,
            )


class TestMaxPool:
    def test_feedforward_config_with_list_of_ints(self):
        config = FeedForwardConfig(
            network_structure=[10, 20, 30],
            classes=("class1", "class2"),
            hidden_activation=Sigmoid(),
            output_activation=Sigmoid(),
            initializer=HeUniform(),
            random_seed=42,
        )

        layers: list[Trainable] = [layer for layer in config.network_structure]

        assert len(config.network_structure) == 3
        assert all(isinstance(layer, Dense) for layer in config.network_structure)
        assert layers[0].output_dim == 10
        assert layers[1].output_dim == 20
        assert layers[2].output_dim == 30
        assert isinstance(layers[0].activation_function, Sigmoid)
        assert isinstance(layers[1].activation_function, Sigmoid)
        assert isinstance(layers[2].activation_function, Sigmoid)

    def test_feedforward_config_with_list_of_layers(self):
        layers = [
            MaxPool(channels=10, filter_size=(3, 3)),
            MaxPool(channels=20, filter_size=(3, 3)),
            MaxPool(channels=30, filter_size=(3, 3)),
        ]
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

    def test_feedforward_config_with_list_of_tuples(self):
        config = FeedForwardConfig(
            network_structure=[
                MaxPool(channels=10, filter_size=(3, 3)),
                MaxPool(channels=20, filter_size=(3, 3)),
                MaxPool(channels=30, filter_size=(3, 3)),
            ],
            classes=("class1", "class2"),
            hidden_activation=Sigmoid(),
            output_activation=Sigmoid(),
            initializer=HeUniform(),
            random_seed=42,
        )

        layers: list[Trainable] = [layer for layer in config.network_structure]

        assert len(config.network_structure) == 3
        assert all(isinstance(layer, MaxPool) for layer in config.network_structure)
        assert layers[0].input_dim == 10
        assert layers[0].output_dim == 10
        assert layers[1].input_dim == 20
