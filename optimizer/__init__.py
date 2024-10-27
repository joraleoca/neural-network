"""
This module implements various optimization algorithms commonly used in training neural networks.
Each optimizer is implemented as a concrete class inheriting from the abstract
Optimizer base class.

Classes:
    - Optimizer: Abstract base class defining the interface for optimization algorithms
    - SGD: Implementation of Stochastic Gradient Descent optimizer
    - Adam: Implementation of the Adam optimizer

Example:
    >>> import numpy as np
    >>> from optimizer import SGD
    >>>
    >>> # Create optimizer instance
    >>> sgd = SGD(learning_rate=0.01)
    >>>
    >>> # Example gradient
    >>> gradient = np.array([0.1, 0.2, 0.3])
    >>>
    >>> # Update parameters
    >>> params = np.array([1.0, 2.0, 3.0])
    >>> updated_params = sgd.update(params, gradient)
    >>> print(updated_params)  # array([0.999, 1.998, 2.997])
"""

from .optimizer import Optimizer
from .sgd import SGD
from .adam import Adam

__all__ = ["Optimizer", "SGD", "Adam"]
