"""
This module implements the loss functions for the neural network.

Classes:
    Loss: Base class for all loss functions.
    CategoricalCrossentropy: Class for categorical cross-entropy loss function.
"""

from .loss import Loss
from .categorical_crossentropy import CategoricalCrossentropy

__all__ = ["Loss", "CategoricalCrossentropy"]
