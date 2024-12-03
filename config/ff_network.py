from dataclasses import dataclass

from structure import Layer
from activation import ActivationFunction
from initialization import Initializator, HeUniform
from encode import Encoder
from core.loader import ParameterLoader


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

    network_structure: list[Layer] | list[int]
    classes: tuple[str, ...]

    initializer: Initializator | ParameterLoader = HeUniform()

    hidden_activation: ActivationFunction | None = None
    output_activation: ActivationFunction | None = None

    encoder: type[Encoder] | None = None

    random_seed: int | None = None

    def __post_init__(self):
        """Validate initialization parameters and set layers parameters."""

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
