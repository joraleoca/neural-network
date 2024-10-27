"""
This module implements various activation functions commonly used in neural networks.
Each activation function is implemented as a concrete class inheriting from the abstract
FunctionActivation base class.

Classes:
    - FunctionActivation: Abstract base class defining the interface for activation functions
    - Relu: Implementation of the Rectified Linear Unit activation function
    - LeakyRelu: Implementation of Leaky ReLU with configurable slope for negative values
    - Tanh: Implementation of the Hyperbolic Tangent activation function
    - Softmax: Implementation of the Softmax activation function for multi-class classification
    - Sigmoid: Implementation of the Sigmoid (logistic) activation function

Example:
    >>> import numpy as np
    >>> from activation import Relu
    >>>
    >>> # Create activation function instance
    >>> relu = Relu()
    >>>
    >>> # Apply activation to data
    >>> data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> activated = relu.activate(data)
    >>> print(activated)  # array([0., 0., 0., 1., 2.])
"""

from .activation import FunctionActivation
from .relu import Relu
from .leaky_relu import LeakyRelu
from .sigmoid import Sigmoid
from .tanh import Tanh
from .softmax import Softmax

__all__ = ["FunctionActivation", "Relu", "LeakyRelu", "Sigmoid", "Tanh", "Softmax"]
