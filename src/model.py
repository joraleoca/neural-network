from abc import ABC, abstractmethod
from typing import Callable
from pathlib import Path

import numpy as np

from . import constants as c
from .tensor import Tensor


class BaseModel(ABC):
    """
    Base class for all network models.
    """

    __slots__ = "layers"

    layers: list[Callable[[Tensor], Tensor]]

    @abstractmethod
    def __call__(self, inputs: Tensor[np.floating]) -> Tensor[np.floating]:
        """
        Perform a forward pass through the neural network.

        Args:
            inputs (Tensor[np.floating]): The input data for the forward pass.
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

    # TODO: Fix this method with the new structure
    def store_params(self, file: str | Path = c.FILE_NAME) -> None:
        """
        Store the weights and biases of the neural network to a file.

        Args:
            file (str | Path): The file path to store the weights and biases.
        """
        kwds = {}

        structure = []

        for i, layer in enumerate(self.layers):
            structure.append(f"{layer.__class__.__name__} {i}")

            layer_data = layer.data_to_store()

            for key, val in layer_data.items():
                kwds[f"{key} {i}"] = val

        kwds[c.STRUCTURE_STR] = np.array(structure, dtype=str)

        np.savez(file, **kwds)

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
