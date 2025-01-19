from copy import deepcopy
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

from . import constants as c
from .config import FeedForwardConfig, TrainingConfig
from .core import (
    Tensor,
    op,
)
from .structure import Layer, Trainable
from .loss import CategoricalCrossentropy
from .encode import Encoder


class NeuralNetwork:
    """
    NeuralNetwork class for building and training a neural network.
    """

    __slots__ = "layers", "encoder", "classes", "_trainable_layers"

    layers: list[Layer]

    encoder: Encoder
    classes: tuple[str, ...]

    # Used to store the trainable layers of the network for optimization
    _trainable_layers: list[Trainable]

    def __init__(self, config: FeedForwardConfig) -> None:
        """
        Initializes the neural network with the given config.

        Args:
            config (NeuralNetworkConfig): Configuration for the neural network.
        """
        self.layers = config.network_structure
        self.classes = config.classes

        if config.encoder:
            self.encoder = config.encoder(self.classes)

        self._trainable_layers = [
            layer for layer in self.layers if isinstance(layer, Trainable)
        ]

    @property
    def num_hidden_layers(self) -> int:
        """
        Returns the number of hidden layers in the neural network.

        Returns:
            int: The number of hidden layers in the neural network.
        """
        return len(self.layers) - 2

    @property
    def params(self) -> list[tuple[Tensor[np.floating], Tensor[np.floating]]]:
        """
        Returns the weights and biases of the neural network for layers.

        Returns:
            list[tuple[Tensor, Tensor]]: The weights and biases of the neural network.
        """
        return [(layer.weights, layer.biases) for layer in self._trainable_layers]

    def forward_pass(self, inputs: Tensor[np.floating]) -> Tensor[np.floating]:
        """
        Perform a forward pass through the neural network.

        Args:
            inputs (Tensor[np.floating]): The input data for the forward pass.
        Returns:
            Tensor[np.floating]: The output of the neural network after the forward pass.
        """
        if not isinstance(inputs, Tensor):
            inputs = Tensor(inputs)

        return self._forward_pass(inputs)

    def _forward_pass(
        self,
        inputs: Tensor[np.floating],
    ) -> Tensor[np.floating]:
        """
        Perform a forward pass through the neural network.

        Args:
            inputs (Tensor[np.floating]): The input data for the neural network.
        Returns:
            Tensor[np.floating]: The output of each layer of the neural network.
        """
        last_output = inputs

        for layer in self.layers:
            last_output = layer.forward(last_output)

        return last_output

    @staticmethod
    def _set_data_requires_grad(data: list[tuple[Tensor[float] | ArrayLike, str]]) -> None:
        """
        Set the requires_grad attribute to True for the input data,
        also convert all data to Tensor if it is not already.\n
        This operation is done in-place.

        Args:
            data (list[tuple[Tensor[float] | ArrayLike[float], str]]): The input data to be set.
        """
        for i, item in enumerate(data):
            if isinstance(item[0], Tensor):
                item[0].requires_grad = True
            else:
                data[i] = (Tensor(item[0], requires_grad=True), item[1])

    def train(
        self,
        data_train: list[tuple[Tensor | ArrayLike, str]],
        data_evaluate: list[tuple[Tensor | ArrayLike, str]],
        *,
        config: TrainingConfig,
    ) -> None:
        """
        Trains the neural network using the provided training and evaluation data.

        Args:
            data_train (list[tuple[Tensor, str]]): The training data, where each element is a
                tuple containing an input array and its corresponding label.
            data_evaluate (list[tuple[Tensor, str]]): The evaluation data, where each element is a
                tuple containing an input array and its corresponding label.
            config (TrainingConfig, optional): Configuration for training.

        Notes:
            - The method implements early stopping based on the validation loss.
            - The best weights and biases are stored and restored if early stopping is triggered.
            - If `config.debug` is True, training and evaluation metrics are printed and stored for plotting.
            - If `config.store` is True, the trained weights and biases are saved to a file.
        """
        encoder_class = config.loss.encoder()
        if not hasattr(self, "encoder") or not isinstance(self.encoder, encoder_class):
            self.encoder = encoder_class(self.classes)

        # Set requires_grad to True for all input data
        self._set_data_requires_grad(data_train)
        self._set_data_requires_grad(data_evaluate)

        for layer in self._trainable_layers:
            layer.requires_grad = True

        last_epoch_loss = float("inf")
        patience_counter = 0

        metrics = {"losses": [], "train_acc": [], "test_acc": []}

        best_layers = []
        best_accuracy = 0.0

        for epoch in range(config.epochs + 1):
            losses = self._backward(data_train, config)
            current_epoch_loss = np.mean(losses)

            val_accuracy = self.evaluate(data_evaluate)

            if val_accuracy >= best_accuracy:
                best_accuracy = val_accuracy
                best_layers = deepcopy(self.layers)

            if current_epoch_loss < last_epoch_loss - config.min_delta:
                patience_counter = 0
                last_epoch_loss = current_epoch_loss
            else:
                patience_counter += 1

            if patience_counter >= config.patience_stop:
                print(f"Early stopping at epoch {epoch} due to no improvement in loss.")
                break

            config.lr.update(epoch)

            if config.debug:
                train_accuracy = self.evaluate(data_train)
                # Store metrics for plotting
                metrics["losses"].append(current_epoch_loss)
                metrics["test_acc"].append(val_accuracy)
                metrics["train_acc"].append(train_accuracy)
                print(
                    f"Epoch {epoch}, "
                    f"Accuracy: {val_accuracy:.4f}, "
                    f"Loss: {current_epoch_loss:.4f}, "
                    f"train_acc: {train_accuracy:.4f}"
                )

        self.layers = best_layers

        for layer in self._trainable_layers:
            layer.requires_grad = False

        if config.store:
            self.store_params()

        if config.debug:
            self._plot_metrics_train(**metrics)

    def _backward(
        self, data: list[tuple[Tensor, str]], config: TrainingConfig
    ) -> list[float]:
        """
        Perform the backward pass of the neural network, updating weights and biases.

        Args:
            data (list[tuple[Tensor, str]]): A list of tuples where each tuple contains
                an input array and the corresponding expected label.
            config (TrainingConfig): Configuration for training.
        Returns:
            list[float]: The losses of the training.
        """
        data_array = np.array(data, dtype=object)
        n_batches = (len(data_array) + config.batch_size - 1) // config.batch_size
        batches = np.array_split(data_array, n_batches)

        losses: list[float] = []

        for data_batch in batches:
            for inputs, expected_label in data_batch:
                expected = self.encoder(expected_label)
                predicted = self._forward_pass(inputs)

                if isinstance(config.loss, CategoricalCrossentropy):
                    # This approach is used because the current autograd implementation does not support
                    # the computation of the Jacobian matrix, which is necessary for calculating the gradient.
                    # TODO: Implement the Jacobian matrix in the autograd system and remove this workaround.
                    # This approach is functional until the Jacobian matrix support is added.
                    loss = op.cce(predicted, expected)
                else:
                    loss = config.loss(expected, predicted)

                losses.append(float(loss.item()))

                loss.backward()

                for layer in self._trainable_layers:
                    layer.backward()

            config.optimizer(config.lr.learning_rate, layers=self._trainable_layers)

            for layer in self._trainable_layers:
                layer.clear_params_grad()

        return losses

    def evaluate(self, data: list[tuple[Tensor, str]]) -> float:
        """
        Evaluate the performance of the neural network on a given dataset.

        Args:
            data (list[tuple[Tensor, str]]): A list of tuples where each tuple contains an input
                             array and the corresponding label.
        Returns:
            float: The accuracy of the neural network on the provided dataset, calculated as the
               proportion of correctly classified inputs.
        """

        correct = sum(
            self.encoder.decode(self.forward_pass(input_)) == label
            for input_, label in data
        )
        return correct / len(data)

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
    def _plot_metrics_train(
        losses: list[float],
        test_acc: list[float],
        train_acc: list[float],
    ) -> None:
        """
        Plot the training metrics including loss, training accuracy, and test accuracy.

        Args:
            losses (list[float]): List of loss values for each epoch.
            test_acc (list[float]): List of test accuracy values for each epoch.
            train_acc (list[float]): List of training accuracy values for each epoch.
        """
        epochs = range(len(losses))

        plt.figure(figsize=(12, 4))

        # Plot loss
        plt.subplot(1, 3, 1)
        plt.plot(epochs, losses, label="Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()

        # Plot test accuracy
        plt.subplot(1, 3, 2)
        plt.plot(epochs, test_acc, label="Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Test Accuracy")
        plt.legend()

        # Plot training accuracy
        plt.subplot(1, 3, 3)
        plt.plot(epochs, train_acc, label="Training Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()
