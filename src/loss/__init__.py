"""
This module implements the loss functions for the neural network.
"""

from .loss import Loss
from .categorical_crossentropy import CategoricalCrossentropy
from .binary_crossentropy import BinaryCrossentropy

__all__ = "Loss", "CategoricalCrossentropy", "BinaryCrossentropy"
