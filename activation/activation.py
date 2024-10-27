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
