"""
This module implements various encoding techniques for neural network preprocessing.
Classes:
    Encoder: A base class for different encoding strategies.
    OneHotEncoder: A class for one-hot encoding.
    BinaryEncoder: A class for binary encoding.
"""

from .encoder import Encoder
from .one_hot_encode import OneHotEncoder
from .binary_encode import BinaryEncoder

__all__ = ["Encoder", "OneHotEncoder", "BinaryEncoder"]
