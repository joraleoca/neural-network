import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np

from src.structure import Layer

from . import constants as c
from .tensor import Tensor


class BaseModel(Layer, ABC):
    """
    Base class for all network models.
    """

    __slots__ = "layers"

    layers: list[Callable[[Tensor], Tensor]]

    @abstractmethod
    def __call__(self, *args: Tensor[np.floating]) -> Tensor[np.floating]:
        """
        Perform a forward pass through the neural network.

        Args:
            args (Tensor[np.floating]): The input data for the forward pass.
        Returns:
            Tensor[np.floating]: The output of the neural network after the forward pass.
        """
        pass

    def parameters(self) -> list[Tensor[np.floating]]:
        """
        Returns all the parameters of the module.

        Returns:
            list[Tensor]: The parameters of the neural network.
        """
        return [param for layer in self.layers if hasattr(layer, "parameters") for param in layer.parameters()]

    # TODO: Make a better implementation of the store_params and load_params methods, this is just a workaround, not prioritary
    def store_params(self, file: str | Path = c.FILE_NAME) -> None:
        """
        Store the weights and biases of the neural network to a file.

        Args:
            file (str | Path): The file path to store the weights and biases.
        """
        np.save(file, np.array(self, dtype=object))

    def load_params(self, path: str | os.PathLike | Path = c.FILE_NAME):
        if not isinstance(path, Path):
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File {path} not found.")
        if path.suffix != ".npy":
            raise ValueError(f"Path must be a numpy .npz file. Got {path.suffix}")
        if not os.access(path, os.R_OK):
            raise ValueError(f"File {path} is not readable.")

        try:
            self.layers = np.load(path, allow_pickle=True).item().layers
        except Exception as e:
            raise RuntimeError(f"Error loading parameters: {e}")

    @staticmethod
    def plot_metrics(
        losses: list[float],
        test_acc: list[float],
    ) -> None:
        """
        Plot the training metrics, loss and training accuracy.

        Args:
            losses (list[float]): List of loss values for each epoch.
            test_acc (list[float]): List of test accuracy values for each epoch.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        # Plot loss
        plt.subplot(1, 3, 1)
        plt.plot(len(losses), losses, label="Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()

        # Plot test accuracy
        plt.subplot(1, 3, 2)
        plt.plot(len(test_acc), test_acc, label="Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Test Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig("training_metrics.png")
