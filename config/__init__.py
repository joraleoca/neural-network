"""
This module initializes the necessary configuration classes used in neural networks.

Classes:
    - FeedForwardConfig: Configuration for a feed-forward neural network
    - TrainingConfig: Configuration for training parameters
"""

from .training_config import TrainingConfig
from .ff_network import FeedForwardConfig

__all__ = [
    "FeedForwardConfig",
    "TrainingConfig",
]
