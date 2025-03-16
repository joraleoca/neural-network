"""
This module implements the basic operations that can be applied to functions.
"""

from .max import Max
from .min import Min
from .mean import Mean
from .sum import Sum
from .tanh import Tanh
from .log import Log
from .reshape import Reshape
from .transpose import Transpose
from .flatten import Flatten
from .ravel import Ravel
from .expand_dims import ExpandDims
from .round import Round
from .pad import Pad
from .stack import Stack
from .cce import CategoricalCrossentropy
from .as_strided import As_Strided
from .argmax import Argmax

__all__ = (
    "Max",
    "Min",
    "Mean",
    "Sum",
    "Tanh",
    "Log",
    "Reshape",
    "Transpose",
    "Flatten",
    "Ravel",
    "ExpandDims",
    "Round",
    "Pad",
    "Stack",
    "CategoricalCrossentropy",
    "As_Strided",
    "Argmax",
)
