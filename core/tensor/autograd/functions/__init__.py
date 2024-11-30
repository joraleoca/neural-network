"""
This module implements the basic operations that can be applied to functions.

Classes:
    Max: Represents the max operation for autograd.
    Sum: Represents the sum operation for autograd.
    Tanh: Represents the tanh operation for autograd.
    Log: Represents the log operation for autograd.
    Reshape: Represents the reshape operation for autograd.
    Transpose: Represents the transpose operation for autograd.
    Round: Represents the round operation for autograd.
    CategoricalCrossentropy: Represents the categorical crossentropy softmax operation for autograd.
"""

from .max import Max
from .sum import Sum
from .tanh import Tanh
from .log import Log
from .reshape import Reshape
from .transpose import Transpose
from .round import Round
from .cce import CategoricalCrossentropy

__all__ = [
    "Max",
    "Sum",
    "Tanh",
    "Log",
    "Reshape",
    "Transpose",
    "Round",
    "CategoricalCrossentropy",
]
