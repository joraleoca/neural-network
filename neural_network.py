from typing import Final, Any
from copy import deepcopy
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from activation import FunctionActivation
from loss import Loss
from encode import Encoder
from optimizer import Optimizer
from config import NeuralNetworkConfig, TrainingConfig
from regularization import Dropout
from core import ParameterLoadError
from core.constants import FILE_NAME, WEIGHT_PREFIX, BIAS_PREFIX, EPSILON


class NeuralNetwork:
    """
    NeuralNetwork class for building and training a neural network.
    """

    __slots__ = (
        "weights",
        "biases",
        "num_hidden_layers",
        "hidden_activation",
        "output_activation",
        "loss",
        "encoder",
        "optimizer",
        "batch_size",
        "dropout",
        "params_file",
    )

    weights: list[NDArray[np.floating[Any]]]
    biases: list[NDArray[np.floating[Any]]]

    num_hidden_layers: Final[int]

    # Function activators
    hidden_activation: FunctionActivation
    output_activation: FunctionActivation

    loss: Loss
    encoder: Encoder

    # Regularization
    optimizer: Optimizer
    batch_size: int
    dropout: Dropout | None

    params_file: str | Path

    MAX_DELTA_NORM: Final[float] = 5.0

    def __init__(self, config: NeuralNetworkConfig) -> None:
        """
        Initializes the neural network with the given config.
        Args:
            config (NeuralNetworkConfig): Configuration for the neural network.
        """
        self.num_hidden_layers = (
            len(config.network_structure) - 2
        )  # First and last layer
        self.hidden_activation = config.hidden_activation
        self.output_activation = config.output_activation
        self.loss = config.loss
        self.encoder = config.loss.encoder()(config.classes)
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        self.dropout = config.dropout
        self.params_file = FILE_NAME

        # Parameters initialization
        if not config.loader:
            self.weights = config.initializator.initializate(
                config.network_structure, rng=np.random.default_rng(config.random_seed)
            )
            self.biases = [
                np.zeros((config.network_structure[i + 1], 1))
                for i in range(len(config.network_structure) - 1)
            ]
        else:
            self.params_file = config.loader.path
            try:
                self.weights, self.biases, expected_hidden_layers = config.loader.load()
            except ParameterLoadError as e:
                print(f"Error loading parameters: {e}")
                return

            if self.num_hidden_layers <= 0:
                self.num_hidden_layers = expected_hidden_layers
            elif self.num_hidden_layers != expected_hidden_layers:
                raise ParameterLoadError(
                    f"Expected {expected_hidden_layers} hidden layers, but got {self.num_hidden_layers}"
                )

    @property
    def classes(self) -> tuple[str, ...]:
        """
        Returns the classes recognized by the neural network.

        Returns:
            tuple[str, ...]: A tuple containing the class labels.
        """
        return self.encoder.classes

    def forward_pass(
        self, inputs: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """
        Perform a forward pass through the neural network.
        Args:
            inputs (NDArray[np.floating[Any]]): The input data for the forward pass.
        Returns:
            NDArray[np.floating[Any]]: The output of the neural network after the forward pass.
        """
        if len(inputs) != self.weights[0].shape[1]:
            raise ValueError(
                f"Input size {len(inputs)} does not match expected size {self.weights[0].shape[1]}"
            )

        output, _ = self._forward_pass(inputs)
        return output[-1]

    def _forward_pass(
        self,
        inputs: NDArray[np.floating[Any]],
        training: bool = False,
    ) -> tuple[list[NDArray[np.floating[Any]]], list[NDArray[np.floating[Any]]]]:
        """
        Perform a forward pass through the neural network.
        Args:
            inputs (NDArray[np.floating[Any]]): The input data for the neural network.
            training (bool): Flag indicating if its training or not
        Returns:
            (tuple[list[NDArray[np.floating[Any]]], list[NDArray[np.floating[Any]]]]):
                A tuple containing two lists:
                - The first list contains the outputs of each layer.
                - The second list contains the inputs to each layer.
        """
        layer_inputs = []
        layer_outputs = [inputs.reshape(-1, 1)]

        for i in range(self.num_hidden_layers + 1):
            z = (self.weights[i] @ layer_outputs[-1]) + self.biases[i]
            layer_inputs.append(z)

            if i != self.num_hidden_layers:
                activation = self.hidden_activation.activate(z)
                if self.dropout and training:
                    self.dropout.drop(activation)
            else:
                activation = self.output_activation.activate(z)

            layer_outputs.append(activation)

        return layer_outputs, layer_inputs

    def _backwards(
        self, lr: float, data: list[tuple[np.ndarray, str]]
    ) -> list[np.floating]:
        """
        Perform the backward pass of the neural network, updating weights and biases.
        Args:
            data (list[tuple[np.ndarray, str]]): A list of tuples where each tuple contains
                an input array and the corresponding expected label.
        Returns:
            list[np.floating]: The losses of the training.
        """
        data_array = np.array(data, dtype=object)
        n_batches = (len(data_array) + self.batch_size - 1) // self.batch_size
        batches = np.array_split(data_array, n_batches)

        losses = []

        for data_batch in batches:
            weights_batch_gradient = [np.zeros_like(w) for w in self.weights]
            biases_batch_gradient = [np.zeros_like(b) for b in self.biases]

            for inputs, expected_label in data_batch:
                expected = self.encoder(expected_label)
                outputs_layers, inputs_layers = self._forward_pass(
                    inputs, training=True
                )
                predicted = outputs_layers[-1]

                losses.append(self.loss(expected, predicted))

                gradient = [self.loss.gradient(expected, predicted)]

                for i in reversed(range(len(self.weights) - 1)):
                    gradient.append(
                        (self.weights[i + 1].T @ gradient[-1])
                        * self.hidden_activation.derivative(inputs_layers[i])
                    )

                # Normalize global batch gradient by dividing by its norm (L2 norm)
                global_norm = np.sqrt(sum(np.linalg.norm(d) ** 2 for d in gradient))
                if global_norm > self.MAX_DELTA_NORM:
                    for i, d in enumerate(gradient):
                        gradient[i] = np.multiply(
                            d, self.MAX_DELTA_NORM / (global_norm + EPSILON)
                        )

                gradient.reverse()

                for i, g in enumerate(gradient):
                    weights_batch_gradient[i] += g @ outputs_layers[i].T
                    biases_batch_gradient[i] += g

            batch_size = len(data_batch)
            weights_batch_gradient = [w / batch_size for w in weights_batch_gradient]
            biases_batch_gradient = [b / batch_size for b in biases_batch_gradient]

            # Update the parameters based on the gradient
            self.optimizer.optimize_weights(
                lr,
                weights_batch_gradient,
                weights=self.weights,
            )
            self.optimizer.optimize_biases(
                lr,
                biases_batch_gradient,
                biases=self.biases,
            )

        return losses

    def train(
        self,
        data_train: list[tuple[np.ndarray, str]],
        data_evaluate: list[tuple[np.ndarray, str]],
        config: TrainingConfig = TrainingConfig(),
    ) -> None:
        """
        Trains the neural network using the provided training and evaluation data.

        Args:
            data_train (list[tuple[np.ndarray, str]]): The training data, where each element is a
                tuple containing an input array and its corresponding label.
            data_evaluate (list[tuple[np.ndarray, str]]): The evaluation data, where each element is a
                tuple containing an input array and its corresponding label.
            config (TrainingConfig, optional): Configuration for training.

        Notes:
            - The method implements early stopping based on the validation loss.
            - The best weights and biases are stored and restored if early stopping is triggered.
            - If `config.debug` is True, training and evaluation metrics are printed and stored for plotting.
            - If `config.store` is True, the trained weights and biases are saved to a file.
        """
        last_epoch_loss = float("inf")
        patience_counter = 0

        metrics = {"losses": [], "train_acc": [], "test_acc": []}

        best_weights = []
        best_biases = []
        best_accuracy = 0.0

        for epoch in range(config.epochs + 1):
            loss = self._backwards(config.lr.learning_rate, data_train)
            current_epoch_loss = np.mean(loss)

            val_accuracy = self.evaluate(data_evaluate)

            if val_accuracy >= best_accuracy:
                best_accuracy = val_accuracy
                best_weights = deepcopy(self.weights)
                best_biases = deepcopy(self.biases)

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

        self.weights = best_weights
        self.biases = best_biases

        if config.store:
            kwds = {f"{WEIGHT_PREFIX}{i}": w for i, w in enumerate(self.weights)}
            kwds.update({f"{BIAS_PREFIX}{i}": b for i, b in enumerate(self.biases)})

            np.savez(self.params_file, **kwds)

        if config.debug:
            self._plot_metrics_train(**metrics)

    def evaluate(self, data: list[tuple[np.ndarray, str]]) -> float:
        """
        Evaluate the performance of the neural network on a given dataset.

        Args:
            data (list[tuple[np.ndarray, str]]): A list of tuples where each tuple contains an input
                             array and the corresponding label.

        Returns:
            float: The accuracy of the neural network on the provided dataset, calculated as the
               proportion of correctly classified inputs.
        """

        def simplify_array(arr: NDArray):
            return arr.item() if arr.size == 1 else arr

        correct = sum(
            self.encoder.decode(simplify_array(self.forward_pass(input))) == label
            for input, label in data
        )
        return correct / len(data)

    def _plot_metrics_train(
        self,
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
