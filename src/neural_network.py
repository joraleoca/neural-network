from copy import deepcopy
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike

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
    

    def _prepare_data_for_training(
        self, data: Iterable[Tensor | ArrayLike], expected: Iterable[str]
    ) -> tuple[list[Tensor], list[Tensor]]:
        """
        Prepare the data for training by encoding the labels and converting the data to Tensors.

        Args:
            data (Iterable[Tensor | ArrayLike]): The input data for training.
            expected (Iterable[str]): The expected labels for the input data.
        Returns:
            tuple[list[Tensor], list[Tensor]]: A tuple containing the input data as Tensors and the encoded labels.
        """
        data_ = []

        for arr in data:
            if isinstance(arr, Tensor):
                arr.requires_grad = True
                data_.append(arr)
            else:
                data_.append(Tensor(arr, requires_grad=True))

            if isinstance(self.layers[0], Convolution) and data_[-1].ndim == 2:
                data_[-1] = op.expand_dims(data_[-1], axis=0)

        out_expected = [self.encoder.encode(label) for label in expected]

        return data_, out_expected

    def _batch_data(
        self, data: list[Tensor], expected: list[Tensor], batch_size: int
    ) -> list[tuple[Tensor, Tensor]]:
        """
        Batch the data for training.

        Args:
            data (list[Tensor]): The input data for training.
            expected (list[Tensor]): The expected labels for the input data.
            batch_size (int): The number of batches to create.
        Returns:
            list[tuple[Tensor, Tensor]]: A list of tuples, where each tuple contains:
                - A Tensor representing the input data for a batch.
                - A Tensor representing the corresponding labels for the batch.
        """
        if batch_size >= len(data):
            return [(op.compose(data), op.compose(expected))]

        i = 0
        out = []

        while i + batch_size < len(data):
            out.append(
                (
                    op.compose(data[i : i + batch_size]),
                    op.compose(expected[i : i + batch_size]),
                )
            )
            i += batch_size

        if i < len(data):
            out.append((op.compose(data[i:]), op.compose(expected[i:])))

        return out


    def train(
        self,
        data_train: Iterable[Tensor | ArrayLike],
        expected_train: Iterable[str],
        data_evaluate: Iterable[Tensor | ArrayLike],
        expected_evaluate: Iterable[str],
        *,
        config: TrainingConfig,
    ) -> None:
        """
        Trains the neural network using the provided training and evaluation data.

        Args:
            data_train (Iterable[Tensor | ArrayLike]): The training data for the neural network.
            expected_train (Iterable[str]): The expected labels for the training data.
            data_evaluate (Iterable[Tensor | ArrayLike]): The evaluation data for the neural network.
            expected_evaluate (Iterable[str]): The expected labels for the evaluation data.
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


        data_train, expected_train_encoded = self._prepare_data_for_training(data_train, expected_train)
        data_evaluate, _ = self._prepare_data_for_training(data_evaluate, expected_evaluate)

        for layer in self._trainable_layers:
            layer.requires_grad = True

        last_epoch_loss = float("inf")
        patience_counter = 0

        metrics = {"losses": [], "train_acc": [], "test_acc": []}

        best_layers = []
        best_trainable_layers = []
        best_accuracy = 0.0

        batches = self._batch_data(data_train, expected_train_encoded, config.batch_size)

        for epoch in range(config.epochs + 1):
            losses = self._backward(batches, config, (data_evaluate, expected_evaluate))
            current_epoch_loss = np.mean(losses)

            val_accuracy = self.evaluate(data_evaluate, expected_evaluate)

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
                # Store metrics for plotting
                metrics["losses"].append(current_epoch_loss)
                metrics["test_acc"].append(val_accuracy)
                print(
                    f"Epoch {epoch}, "
                    f"Accuracy: {val_accuracy:.4f}, "
                    f"Loss: {current_epoch_loss:.4f}, "
                )

        self.layers = best_layers
        self._trainable_layers = best_trainable_layers

        for layer in self._trainable_layers:
            layer.requires_grad = False

        if config.store:
            self.store_params()

        if config.debug:
            self._plot_metrics_train(**metrics)

    def _backward(self, batches: list[tuple[Tensor, Tensor]], config: TrainingConfig, eval) -> list[float]:
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

            losses.extend(loss.item() for loss in batch_loss)

            batch_loss.backward()

            config.optimizer(config.lr.learning_rate, layers=self._trainable_layers)

            for layer in self._trainable_layers:
                layer.clear_params_grad()

        return losses

    def evaluate(self, data: Iterable[Tensor], expected: Iterable[str]) -> float:
        """
        Evaluate the performance of the neural network on a given dataset.

        Args:
            data (Iterable[Tensor]): The input data for evaluation.
            expected (Iterable[str]): The expected labels for the input data.

        Returns:
            float: The accuracy of the neural network on the provided dataset, calculated as the
               proportion of correctly classified inputs.
        """
        correct = 0
        total = 0

        for input_, label in zip(data, expected):
            total += 1

            if self.encoder.decode(self.forward_pass(input_)) == label:
                correct += 1

        return correct / total

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
        import matplotlib.pyplot as plt

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

        plt.tight_layout()
        plt.savefig("training_metrics.png")
