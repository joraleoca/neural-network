import os
from pathlib import Path

import numpy as np

from src.structure import Layer

from . import constants as c


# TODO: Make a better implementation of the store_params and load_params methods, this is just a workaround, not prioritary
def store_params(layer: Layer, file: str | Path = c.FILE_NAME) -> None:
    """
    Store the weights and biases of the neural network to a file.

    Args:
        file (str | Path): The file path to store the weights and biases.
    """
    np.save(file, np.array(layer, dtype=object))


def load_params(layer: Layer, path: str | os.PathLike | Path = c.FILE_NAME) -> Layer:
    if not isinstance(path, Path):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File {path} not found.")
    if path.suffix != ".npy":
        raise ValueError(f"Path must be a numpy .npz file. Got {path.suffix}")
    if not os.access(path, os.R_OK):
        raise ValueError(f"File {path} is not readable.")

    try:
        layer = np.load(path, allow_pickle=True).item()
    except Exception as e:
        raise RuntimeError(f"Error loading parameters: {e}")

    if not isinstance(layer, Layer):
        raise TypeError(f"Loaded object is not a Layer. Got {type(layer)}")

    return layer


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
