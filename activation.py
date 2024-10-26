"""
activation.py

This module implements various activation functions commonly used in neural networks.
Each activation function is implemented as a concrete class inheriting from the abstract
FunctionActivation base class.

Classes:
    FunctionActivation: Abstract base class defining the interface for activation functions
    Relu: Implementation of the Rectified Linear Unit activation function
    LeakyRelu: Implementation of Leaky ReLU with configurable slope for negative values
    Tanh: Implementation of the Hyperbolic Tangent activation function
    Softmax: Implementation of the Softmax activation function for multi-class classification
    Sigmoid: Implementation of the Sigmoid (logistic) activation function

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

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class FunctionActivation(ABC):
    """
    Abstract base class defining the interface for neural network activation functions.

    All activation functions must implement the activate() and derivative() methods.
    Input validation is provided through the protected _validate_input() method.
    """

    @abstractmethod
    def activate(self, data: NDArray) -> NDArray:
        """
        Apply the activation function to the input data.

        Args:
            data: Input values to be transformed by the activation function.
                 Expected to be a numpy array of any shape.

        Returns:
            Transformed input values after applying the activation function.

        Raises:
            ValueError: If input is not a numpy array.
        """
        pass

    @abstractmethod
    def derivative(self, data: NDArray) -> NDArray:
        """
        Compute the derivative of the activation function at the given input values.

        Args:
            data: Input values at which to compute the derivative.
                 Expected to be a numpy array of any shape.

        Returns:
            Derivative values of the activation function at the input points.

        Raises:
            ValueError: If input is not a numpy array.
        """
        pass

    def _validate_input(self, data: NDArray) -> None:
        """
        Validate that the input data is a numpy array.

        Args:
            data: Input to validate.

        Raises:
            ValueError: If input is not a numpy array.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array.")


class Relu(FunctionActivation):
    """
    Rectified Linear Unit (ReLU) activation function.

    Computes the function: f(x) = max(0, x)

    The derivative is:
        f'(x) = 1 if x > 0
        f'(x) = 0 if x ≤ 0
    """

    def activate(self, data: NDArray) -> NDArray:
        self._validate_input(data)
        return np.maximum(0, data)

    def derivative(self, data: NDArray) -> NDArray:
        self._validate_input(data)
        return np.where(data > 0, 1, 0)


class LeakyRelu(FunctionActivation):
    """
    Leaky Rectified Linear Unit activation function.

    Computes the function:
        f(x) = x if x > 0
        f(x) = αx if x ≤ 0
    where α is a small positive constant.

    Args:
        alpha: Slope for negative values. Defaults to 0.01.
    """

    ALPHA: float

    def __init__(self, *, alpha: float = 0.01) -> None:
        """
        Initialize LeakyRelu with given alpha parameter.

        Args:
            alpha: Slope for negative values. Must be positive.
        """
        self.ALPHA = alpha

    def activate(self, data: NDArray) -> NDArray:
        self._validate_input(data)
        return np.where(data > 0, data, self.ALPHA * data)

    def derivative(self, data: NDArray) -> NDArray:
        self._validate_input(data)
        return np.where(data > 0, 1, self.ALPHA)


class Tanh(FunctionActivation):
    """
    Hyperbolic tangent activation function.

    Computes the function: f(x) = tanh(x)

    The derivative is: f'(x) = 1 - tanh²(x)
    """

    def activate(self, data: NDArray) -> NDArray:
        self._validate_input(data)
        return np.tanh(data)

    def derivative(self, data: NDArray) -> NDArray:
        self._validate_input(data)
        return 1 - (np.tanh(data) ** 2)


class Softmax(FunctionActivation):
    """
    Softmax activation function for multi-class classification.

    Computes the function: f(x_i) = exp(x_i) / Σ(exp(x_j))
    where the sum is over all elements j.

    Note:
        The derivative is not implemented as it's typically combined
        with cross-entropy loss for better numerical stability.
    """

    def activate(self, data: NDArray) -> NDArray:
        self._validate_input(data)
        exp_shifted = np.exp(data - np.max(data, axis=0, keepdims=True))
        return exp_shifted / np.sum(exp_shifted, axis=0, keepdims=True)

    def derivative(self, data: NDArray) -> NDArray:
        raise NotImplementedError("Softmax derivative is not implemented")


class Sigmoid(FunctionActivation):
    """
    Sigmoid (logistic) activation function.

    Computes the function: f(x) = 1 / (1 + exp(-x))

    The derivative is: f'(x) = f(x)(1 - f(x))
    """

    def activate(self, data: NDArray) -> NDArray:
        self._validate_input(data)
        return 1 / (1 + np.exp(-data))

    def derivative(self, data: NDArray) -> NDArray:
        self._validate_input(data)
        sigmoid = self.activate(data)
        return sigmoid * (1 - sigmoid)
