from .activation import ActivationFunction
from .relu import Relu
from .leaky_relu import LeakyRelu
from .sigmoid import Sigmoid
from .softmax import Softmax
from .tanh import Tanh

def activation_from_name(name: str) -> type[ActivationFunction]:
    """
    Return the activation function corresponding to the given name.

    Args:
        name (str): The name of the activation function.
    Returns:
        The activation function corresponding to the given name.
    """
    match name:
        case Relu.__name__:
            return Relu
        case LeakyRelu.__name__:
            return LeakyRelu
        case Sigmoid.__name__:
            return Sigmoid
        case Softmax.__name__:
            return Softmax
        case Tanh.__name__:
            return Tanh
        case _:
            raise ValueError(f"Unknown activation function '{name}'")
