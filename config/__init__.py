"""
This module initializes the config package and imports the necessary configuration classes used in neural networks.

Classes:
    - TrainingConfig: Configuration for training parameters
    - NeuralNetworkConfig: Configuration for the neural network
"""

from .training_config import TrainingConfig
from .nn_config import NeuralNetworkConfig

__all__ = ["TrainingConfig", "NeuralNetworkConfig"]
