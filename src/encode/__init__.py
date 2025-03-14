"""
This module implements various encoding techniques for neural network preprocessing.
"""

from .encoder import Encoder
from .one_hot_encode import OneHotEncoder
from .binary_encode import BinaryEncoder

__all__ = "Encoder", "OneHotEncoder", "BinaryEncoder"
