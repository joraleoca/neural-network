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
from .expand_dims import ExpandDims
from .round import Round
from .pad import Pad
from .compose import Compose
from .cce import CategoricalCrossentropy

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
    "ExpandDims",
    "Round",
    "Pad",
    "Compose",
    "CategoricalCrossentropy",
)
