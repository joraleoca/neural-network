from src.structure import Layer, Trainable
from src.activation import ActivationFunction
from src.initialization import Initializer, HeUniform
from src.encode import Encoder
from src.loader import Loader


class FeedForwardConfig:
    """
    Represents the configuration for a feed-forward neural network.\n
    If the initializer is a Loader then the network structure and the classes will be overwritten.\n
    The encoder can be inferred from the loss function in the training configuration.\n

    Attributes:
        network_structure (list[Layer]): A list of layers that compose the neural network.
        classes (tuple): A tuple containing the class labels for the output layer.
        initializer (Initializer | Loader): The initializer for the network parameters.
        hidden_activation (ActivationFunction | None): The activation function for the hidden layers.
        output_activation (ActivationFunction | None): The activation function for the output layer.
        encoder (type[Encoder] | None): The encoder used to encode the input data.
        random_seed (int | None): The random seed for reproducibility.
    """

    network_structure: list[Layer]

    classes: tuple

    initializer: Initializer | Loader = HeUniform()

    hidden_activation: ActivationFunction | None = None
    output_activation: ActivationFunction | None = None

    encoder: type[Encoder] | None = None

    random_seed: int | None = None

    def __init__(
        self,
        network_structure: list[Layer] | None,
        classes: tuple,
        initializer: Initializer | Loader = HeUniform(),
        hidden_activation: ActivationFunction | None = None,
        output_activation: ActivationFunction | None = None,
        encoder: type[Encoder] | None = None,
        random_seed: int | None = None,
    ) -> None:
        self.classes = classes
        self.initializer = initializer
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.encoder = encoder
        self.random_seed = random_seed

        if network_structure:
            self._prepare_network_structure(network_structure)

        if isinstance(self.initializer, Loader):
            self.network_structure = self.initializer.load()

        assert self.network_structure is not None, "No network structure or loader provided"


    def _prepare_network_structure(self, structure: list[Layer]) -> None:
        """Prepare the network structure by setting the activation functions and initializers."""

        assert self.initializer is not None, "Initializer must be provided."

        if not isinstance(structure, list):
            raise TypeError("structure must be a list.")

        if not isinstance(self.initializer, Initializer):
            raise TypeError("initializer must be an Initializer.")

        for layer in structure:
            if not isinstance(layer, Trainable):
                continue

            if layer.activation_function is None:
                if layer is structure[-1]:
                    layer.activation_function = self.output_activation
                else:
                    layer.activation_function = self.hidden_activation

            if layer.initializer is None:
                layer.initializer = self.initializer

            if layer.rng is None:
                layer.rng = self.random_seed

        self.network_structure = structure