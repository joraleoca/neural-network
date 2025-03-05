import pytest

from src.structure import Dense, Convolution, MaxPool, Trainable, Sigmoid, Softmax, Relu
from src.initialization import HeUniform, XavierNormal
from src.encode import Encoder
from src.config.ff_network import FeedForwardConfig


class TestDense:
    def test_feedforward_config_with_list_of_layers(self):
        layers = [
            Dense(features=10),
            Dense(features=20),
            Dense(features=30),
        ]
        config = FeedForwardConfig(
            network_structure=layers,
            classes=("class1", "class2"),
            initializer=HeUniform(),
            random_seed=42,
        )

        assert len(config.network_structure) == 3, f"The network structure should have 3 layers. Got {len(config.network_structure)}"
        assert config.network_structure == layers, f"The network structure should be {layers}. Got {config.network_structure}"

    def test_feedforward_config_with_list_of_tuples(self):
        config = FeedForwardConfig(
            network_structure=[
                Dense((10, 20)),
                Dense((20, 30)),
                Dense((30, 40)),
                Sigmoid(),
            ],
            classes=("class1", "class2"),
            initializer=HeUniform(),
            random_seed=42,
        )

        layers: list[Trainable] = [layer for layer in config.network_structure if isinstance(layer, Trainable)]

        assert len(config.network_structure) == 4, f"The network structure should have 4 layers. Got {len(config.network_structure)}"
        assert all(isinstance(layer, Dense) for layer in layers), f"All trainable layers should be Dense. Got {layers}"
        assert isinstance(config.network_structure[-1], Sigmoid), f"The last layer should be a Sigmoid activation function. Got {config.network_structure[-1]}"
        assert layers[0].input_dim == 10, f"The input dimension of the first layer should be 10. Got {layers[0].input_dim}"
        assert layers[0].output_dim == 20, f"The output dimension of the first layer should be 20. Got {layers[0].output_dim}"
        assert layers[1].input_dim == 20, f"The input dimension of the second layer should be 20. Got {layers[1].input_dim}"
        assert layers[1].output_dim == 30, f"The output dimension of the second layer should be 30. Got {layers[1].output_dim}"
        assert layers[2].input_dim == 30, f"The input dimension of the third layer should be 30. Got {layers[2].input_dim}"
        assert layers[2].output_dim == 40, f"The output dimension of the third layer should be 40. Got {layers[2].output_dim}"

    def test_feedforward_config_with_incomplete_list_of_layers(self):
        with pytest.raises(ValueError):
            FeedForwardConfig(
                network_structure=[
                    Dense(features=10),
                    Dense(features=0),  # Invalid layer with 0 features
                    Dense(features=30),
                ],
                classes=("class1", "class2"),
                initializer=HeUniform(),
                random_seed=42,
            )

    def test_feedforward_config_missing_required_attributes(self):
        with pytest.raises(TypeError):
            FeedForwardConfig(
                network_structure=[Dense(10)],
                initializer=HeUniform(),
                random_seed=42,
            )

    def test_feedforward_config_invalid_types(self):
        with pytest.raises(TypeError):
            FeedForwardConfig(
                network_structure="invalid_type",
                classes=("class1", "class2"),
                initializer=HeUniform(),
                random_seed=42,
            )

    def test_feedforward_config_with_valid_encoder(self):
        class CustomEncoder(Encoder):
            pass

        config = FeedForwardConfig(
            network_structure=[10, 20, 30],
            classes=("class1", "class2"),
            initializer=HeUniform(),
            encoder=CustomEncoder,
            random_seed=42,
        )

        assert config.encoder == CustomEncoder

    def test_feedforward_config_with_different_initializers(self):
        config_he = FeedForwardConfig(
            network_structure=[Dense(10)],
            classes=("class1", "class2"),
            initializer=HeUniform(),
            random_seed=42,
        )

        config_xavier = FeedForwardConfig(
            network_structure=[Dense(10)],
            classes=("class1", "class2"),
            initializer=XavierNormal(),
            random_seed=42,
        )

        assert isinstance(config_he.initializer, HeUniform)
        assert isinstance(config_he.network_structure[0].initializer, HeUniform)
        assert isinstance(config_xavier.initializer, XavierNormal)
        assert isinstance(config_xavier.network_structure[0].initializer, XavierNormal)

    def test_feedforward_config_with_layer_initializer(self):
        config = FeedForwardConfig(
            network_structure=[
                Dense(features=10, initializer=HeUniform()),
                Dense(features=20, initializer=XavierNormal()),
                Dense(features=30),
            ],
            classes=("class1", "class2"),
            random_seed=42,
        )

        layers: list[Trainable] = [layer for layer in config.network_structure] 

        assert isinstance(layers[0]._initializer, HeUniform)
        assert isinstance(layers[1]._initializer, XavierNormal)
        assert isinstance(layers[2]._initializer, HeUniform)


class TestConvultion:
    def test_list_of_layers(self):
        layers = [
            Convolution(channels=10, kernel_shape=(3, 3)),
            Convolution(channels=20, kernel_shape=(3, 3)),
            Convolution(channels=30, kernel_shape=(3, 3)),
        ]
        config = FeedForwardConfig(
            network_structure=layers,
            classes=("class1", "class2"),
            initializer=HeUniform(),
            random_seed=42,
        )

        assert len(config.network_structure) == 3
        assert config.network_structure == layers

    def test_list_of_tuples(self):
        config = FeedForwardConfig(
            network_structure=[
                Convolution((10, 20), kernel_shape=(3, 3)),
                Convolution((20, 30), kernel_shape=(3, 3)),
                Convolution((30, 40), kernel_shape=(3, 3)),
            ],
            classes=("class1", "class2"),
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

    def test_list_of_layers_and_activations(self):
        layers = [
            Convolution(10, kernel_shape=(3, 3)),
            Relu(),
            Convolution(20, kernel_shape=(3, 3)),
            Convolution(20, kernel_shape=(3, 3)),
            Softmax(),
            Convolution(30, kernel_shape=(3, 3)),
        ]
        config = FeedForwardConfig(
            network_structure=layers,
            classes=("class1", "class2"),
            initializer=HeUniform(),
            random_seed=42,
        )

        assert len(config.network_structure) == 6, f"The network structure should have 6 layers. Got {len(config.network_structure)}"
        assert config.network_structure == layers, f"The network structure should be {layers}. Got {config.network_structure}"
        assert isinstance(config.network_structure[1], Relu), f"The second layer should be a Relu activation function. Got {config.network_structure[1]}"
        assert isinstance(config.network_structure[4], Softmax), f"The fifth layer should be a Softmax activation function. Got {config.network_structure[4]}"


    def test_incomplete_list_of_layers(self):
        with pytest.raises(ValueError):
            FeedForwardConfig(
                network_structure=[
                    Convolution(10, (3, 3)),
                    Convolution(0, (3, 3)),  # Invalid layer with 0 channels
                    Convolution(30, (3, 3)),
                ],
                classes=("class1", "class2"),
                initializer=HeUniform(),
                random_seed=42,
            )


class TestMaxPool:
    def test_create(self):
        layer = MaxPool(channels=10, filter_shape=(3, 3), stride=1, padding=0)
        assert layer.channels == 10
        assert layer.filter_shape == (3, 3)
        assert layer.stride == 1
        assert layer.padding == 0

    @pytest.mark.parametrize(
        "channels, filter_shape, stride, padding",
        [
            (-10, (3, 3), 1, 0),
            (20, -3, 1, 0),
            (30, (3, 3), 0, 0),
            (30, (3, 3), -1, 0),
            (30, (3, 3), 1, -1),
        ],
        ids=("negative_channels", "invalid_filter_shape", "zero_stride", "negative_stride", "negative_padding"),
    )
    def test_create_error(self, channels, filter_shape, stride, padding):
        with pytest.raises(ValueError):
            MaxPool(channels=channels, filter_shape=filter_shape, stride=stride, padding=padding)

    def test_creating_network(self):
        layers = [
            MaxPool(channels=10, filter_shape=(3, 3), stride=10, padding=10),
            MaxPool(channels=20, filter_shape=(3, 3)),
            MaxPool(channels=30, filter_shape=(3, 3)),
        ]
        config = FeedForwardConfig(
            network_structure=layers,
            classes=("class1", "class2"),
            initializer=HeUniform(),
            random_seed=42,
        )

        assert len(config.network_structure) == 3
        assert config.network_structure == layers
        assert len(config.network_structure) == 3
        assert all(isinstance(layer, MaxPool) for layer in config.network_structure)
        assert layers[0].input_dim == 10
        assert layers[0].output_dim == 10
        assert layers[0].filter_shape == (3, 3)
        assert layers[0].stride == 10
        assert layers[0].padding == 10
        