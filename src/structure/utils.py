from .layer import Layer
from .trainable import Dense, Convolution
from .regularization import Dropout, Flatten
from .pooling import MaxPool, AveragePool
from .activation import Relu, LeakyRelu, Sigmoid, Softmax, Tanh


def layer_from_name(name: str) -> type[Layer]:
    """
    Creates a layer from the given data.

    Args:
        name (str): The name of the layer.
    Returns:
        Layer: The created layer.
    """
    match name:
        case Dense.__name__:
            return Dense
        case Convolution.__name__:
            return Convolution
        case Dropout.__name__:
            return Dropout
        case Flatten.__name__:
            return Flatten
        case MaxPool.__name__:
            return MaxPool
        case AveragePool.__name__:
            return AveragePool
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
            raise ValueError(f"Unknown layer name: {name}")
