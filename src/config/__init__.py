"""
This module initializes the necessary configuration classes used in neural networks.
"""

from .training_config import TrainingConfig
from .ff_network import FeedForwardConfig

__all__ = "FeedForwardConfig", "TrainingConfig"
