"""
This module initializes the necessary configuration classes used in neural networks.

Classes:
    - NeuralNetworkConfig: Configuration for the neural network
    - TrainingConfig: Configuration for training parameters
"""

from .nn_config import NeuralNetworkConfig
from .training_config import TrainingConfig

__all__ = ["NeuralNetworkConfig", "TrainingConfig"]
