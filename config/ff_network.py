from dataclasses import dataclass

import numpy as np

from structure import Layer
from activation import ActivationFunction
from initialization import Initializator, HeUniform
from encode import Encoder
from core import ParameterLoader, ParameterLoadError


@dataclass(slots=True)
class FeedForwardConfig:
    """
    Represents the configuration for a feed-forward neural network.\n
    If load_parameters is True and the network structure does not have hidden layers or any layer, they will be loaded from the file.\n
    The encoder can be inferred from the loss function in the training configuration.\n

    Attributes:
        network_structure (list[Layer] | list[int]): A list of layers that compose the neural network.
        classes (tuple[str, ...]): A tuple containing the class labels for the output layer.
        initializer (Initializator | ParameterLoader): The initializer for the network parameters.
        hidden_activation (ActivationFunction | None): The activation function for the hidden layers.
        output_activation (ActivationFunction | None): The activation function for the output layer.
        random_seed (int | None): The random seed for reproducibility.
    """

    network_structure: list[Layer] | list[int] | None

    classes: tuple[str, ...]

    initializer: Initializator | ParameterLoader = HeUniform()

    hidden_activation: ActivationFunction | None = None
    output_activation: ActivationFunction | None = None

    encoder: type[Encoder] | None = None

    random_seed: int | None = None

    def __post_init__(self):
        """Validate initialization parameters and set layers parameters."""

        if self.network_structure:
            self._prepare_network_structure()

        if isinstance(self.initializer, ParameterLoader):
            self.load_parameters()

    def _prepare_network_structure(self):
        """Prepare the network structure by setting the activation functions and initializers."""
        if not isinstance(self.network_structure, list):
            raise TypeError("network_structure must be a list.")

        if isinstance(self.network_structure[0], int):
            self.network_structure = [Layer(n) for n in self.network_structure]

        for layer in self.network_structure:
            if layer.activation_function is None:
                if layer is self.network_structure[-1]:
                    layer.activation_function = self.output_activation
                else:
                    layer.activation_function = self.hidden_activation

            if (
                isinstance(self.initializer, Initializator)
                and layer.initializer is None
            ):
                layer.initializer = self.initializer

            if layer.rng is None:
                layer.rng = self.random_seed

    def load_parameters(self):
        """
        Load parameters from a file using the initializer.

        Raises:
            ParameterLoadError: If any error occurs while loading the parameters.
        """
        weights, biases, expected_hidden_layers = self.initializer.load()

        network = bool(self.network_structure)

        if not network:
            self.network_structure = []

        for i, w in enumerate(weights):
            if i < len(self.network_structure):
                layer = self.network_structure[i]

                if hasattr(layer, "in_features") and layer.in_features != w.shape[1]:
                    raise ParameterLoadError(
                        f"Expected {layer.in_features} input features, but got {w.shape[1]}"
                    )

                if layer.out_features != w.shape[0]:
                    raise ParameterLoadError(
                        f"Expected {layer.out_features} output features, but got {w.shape[0]}"
                    )

                layer.in_features = np.prod(w.shape[1:])
                layer.out_features = w.shape[0]
            else:
                layer = Layer(
                    w.shape[::-1], self.hidden_activation, rng=self.random_seed
                )

                if i == len(weights) - 1:
                    layer.activation_function = self.output_activation

                self.network_structure.append(layer)

            layer.weights = w
            layer.biases = biases[i]

        if expected_hidden_layers != len(self.network_structure) - 2:
            raise ParameterLoadError(
                f"Expected {expected_hidden_layers} hidden layers, but got {len(self.network_structure) - 2}"
            )
