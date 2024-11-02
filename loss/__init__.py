"""
This module implements the loss functions for the neural network.

Classes:
    Loss: Base class for all loss functions.
    CategoricalCrossentropy: Class for categorical cross-entropy loss function.
    BinaryCrossentropy: Class for binary cross-entropy loss function.
"""

from .loss import Loss
from .categorical_crossentropy import CategoricalCrossentropy
from .binary_crossentropy import BinaryCrossentropy

__all__ = ["Loss", "CategoricalCrossentropy", "BinaryCrossentropy"]
