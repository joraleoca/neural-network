from dataclasses import dataclass, field

from activation import ActivationFunction
from initialization import Initializator, HeUniform
from loss import Loss, BinaryCrossentropy
from optimizer import Optimizer, SGD
from regularization import Dropout
from core.loader import ParameterLoader


@dataclass(slots=True)
class NeuralNetworkConfig:
    """
    NeuralNetworkConfig is a configuration class for setting up a neural network.
    If load_parameters is True and the network structure does not have hidden layers or any layer, they will be loaded from the file.

    Attributes:
        network_structure (list[int]): A list representing the number of nodes in each layer of the network.
        classes (tuple[str, ...]): A tuple containing the class labels for the output layer.
        hidden_activation (ActivationFunction): The activation function to be used for the hidden layers.
        output_activation (ActivationFunction): The activation function to be used for the output layer.
        loss (Loss): The loss functions.
        initializator (Initializator): The weight initialization strategy.
        optimizer (Optimizer): The optimization algorithm.
        batch_size (int): Size of the batches used for training.
        dropout (Dropout | None): The dropout configuration.
        loader (ParameterLoader | None): The parameters loader from a file.
        random_seed (int | None): The random seed for reproducibility.
    """

    network_structure: list[int]
    classes: tuple[str, ...]

    hidden_activation: ActivationFunction
    output_activation: ActivationFunction

    loss: Loss

    initializator: Initializator = field(default_factory=HeUniform)
    optimizer: Optimizer = field(default_factory=SGD)

    batch_size: int = 1
    dropout: Dropout | None = None

    loader: ParameterLoader | None = None

    random_seed = None

    def __post_init__(self):
        if (not self.loader and not self.network_structure) or any(
            n <= 0 for n in self.network_structure
        ):
            raise ValueError(
                "Invalid network structure. Each layer must have at least one node."
            )
        if self.batch_size < 1:
            raise ValueError(
                f"The batch size must be positive number. Got {self.batch_size}"
            )
        if isinstance(self.loss, BinaryCrossentropy) and len(self.classes) != 2:
            raise ValueError(
                f"The binary cross-entropy only works with 2 classes. Got {len(self.classes)}"
            )
