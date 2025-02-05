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
from .structure import Layer, Trainable, Convolution
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

    def forward_pass(self, input: Tensor[np.floating] | ArrayLike) -> Tensor[np.floating]:
        """
        Perform a forward pass through the neural network.

        Args:
            input (Tensor[np.floating]): The input data for the forward pass.
        Returns:
            Tensor[np.floating]: The output of the neural network after the forward pass.
        """
        if not isinstance(input, Tensor):
            input = Tensor(input)
        return self._forward_pass(input)

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

    def _prepare_data_for_train(
            self, data: list[tuple[Tensor[float] | ArrayLike, str]]
    ) -> list[tuple[Tensor[float], Tensor]]:
        """
        Set the requires_grad attribute to True for the input data,
        Convert all data to Tensor if it is not already.\n
        Encodes labels.\n

        Args:
            data (list[tuple[Tensor[float] | ArrayLike[float], str]]): The input data to be set.
        Returns:
            list[tuple[Tensor[float] | ArrayLike[float], str]]: The data and encoded labels.
        """
        return [(Tensor(item[0], requires_grad=True), self.encoder(item[1])) for item in data]

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
        data_train_encoded = self._prepare_data_for_train(data_train)

        for layer in self._trainable_layers:
            layer.requires_grad = True

        last_epoch_loss = float("inf")
        patience_counter = 0

        metrics = {"losses": [], "train_acc": [], "test_acc": []}

        best_layers = []
        best_trainable_layers = []
        best_accuracy = 0.0

        n_batches = (len(data_train_encoded) + config.batch_size - 1) // config.batch_size
        batches_np = np.array_split(np.array(data_train_encoded, dtype=object), n_batches)
        batches = []

        for data_batch in batches_np:
            data = []
            expected = []
            for inputs, expect in data_batch:
                if isinstance(self.layers[0], Convolution) and inputs.ndim == 2:
                    inputs = op.expand_dims(inputs, axis=0)
                data.append(inputs)
                expected.append(expect)
            batches.append((op.compose(data), Tensor(expected)))

        for epoch in range(config.epochs + 1):
            losses = self._backward(batches, config)
            current_epoch_loss = np.mean(losses)

            val_accuracy = self.evaluate(data_evaluate)

            if val_accuracy >= best_accuracy:
                best_accuracy = val_accuracy
                best_layers = deepcopy(self.layers)
                best_trainable_layers = deepcopy(self._trainable_layers)

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
        self._trainable_layers = best_trainable_layers

        for layer in self._trainable_layers:
            layer.requires_grad = False

        if config.store:
            self.store_params()

        if config.debug:
            self._plot_metrics_train(**metrics)

    def _backward(
        self, batches: list[tuple[Tensor, Tensor]], config: TrainingConfig
    ) -> list[float]:
        """
        Perform the backward pass of the neural network to update weights and biases.

        Args:
            batches (list[tuple[Tensor, Tensor]]): A list of tuples, where each tuple contains:
                - A Tensor representing the input data for a batch.
                - A Tensor representing the corresponding labels for the batch.
            config (TrainingConfig): Configuration parameters for training.
        Returns:
            list[float]: A list of loss values for each batch, representing the error computed
                during the backward pass.
        """
        losses: list[float] = []

        for data, expected in batches:
            predicted = self._forward_pass(data)

            if isinstance(config.loss, CategoricalCrossentropy):
                # This approach is used because the current autograd implementation does not support
                # the computation of the Jacobian matrix, which is necessary for calculating the gradient.
                # TODO: Implement the Jacobian matrix in the autograd system and remove this workaround.
                batch_loss = op.cce(predicted, expected)
            else:
                batch_loss = config.loss(expected, predicted)

            losses.extend([l.item() for l in batch_loss])

            batch_loss.backward()

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
