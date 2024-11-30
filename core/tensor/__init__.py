"""
This module contains the basic primitives.

Modules:
    op: A module containing the basic operations that can be applied to tensors.

Classes:
    Tensor: Represents a tensor.
"""

from .tensor import Tensor
from . import op


__all__ = ["op", "Tensor"]
