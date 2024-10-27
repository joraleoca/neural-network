from dataclasses import dataclass, field

from activation import FunctionActivation
from initialization import Initializator, HeUniform
from optimizer import Optimizer, SGD
from regularization import Dropout


@dataclass(slots=True)
class NeuralNetworkConfig:
    """
    NeuralNetworkConfig is a configuration class for setting up a neural network.
    Attributes:
        network_structure (list[int]): A list representing the number of nodes in each layer of the network.
        classes (tuple[str, ...]): A tuple containing the class labels for the output layer.
        hidden_activation (FunctionActivation): The activation function to be used for the hidden layers.
        output_activation (FunctionActivation): The activation function to be used for the output layer.
        initializator (Initializator): The weight initialization strategy, default is HeUniform.
        optimizer (Optimizer): The optimization algorithm, default is SGD.
        dropout (Dropout | None): The dropout configuration, default is None.
        random_seed (int | None): The random seed for reproducibility, default is None.
    """

    network_structure: list[int]
    classes: tuple[str, ...]
    hidden_activation: FunctionActivation
    output_activation: FunctionActivation

    initializator: Initializator = field(default_factory=HeUniform)
    optimizer: Optimizer = field(default_factory=SGD)
    dropout: Dropout | None = None

    load_parameters: bool = False
    load_file_name: str = "params.npz"
    store_file_name: str = "params.npz"

    random_seed = None

    def __post_init__(self):
        if not self.network_structure or any(n <= 0 for n in self.network_structure):
            raise ValueError(
                "Invalid network structure. Each layer must have at least one node."
            )
        if self.network_structure[-1] != len(self.classes):
            raise ValueError(
                "The network must have the same output nodes as output classes"
            )
