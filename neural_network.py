from typing import Final
from copy import deepcopy
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

from config import NeuralNetworkConfig, TrainingConfig
from core import ParameterLoadError, Tensor, op, constants as c

from activation import ActivationFunction
from loss import Loss, CategoricalCrossentropy
from encode import Encoder
from optimizer import Optimizer
from regularization import Dropout


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

    weights: list[Tensor[np.floating]]
    biases: list[Tensor[np.floating]]

    num_hidden_layers: Final[int]

    # Function activators
    hidden_activation: ActivationFunction
    output_activation: ActivationFunction

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
        self.params_file = c.FILE_NAME

        # Parameters initialization
        if not config.loader:
            self.weights = config.initializator.initialize(
                config.network_structure, rng=np.random.default_rng(config.random_seed)
            )
            self.biases = [
                op.zeros((config.network_structure[i + 1], 1), requires_grad=True)
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

    def forward_pass(self, inputs: Tensor[np.floating]) -> Tensor[np.floating]:
        """
        Perform a forward pass through the neural network.
        Args:
            inputs (Tensor[np.floating]): The input data for the forward pass.
        Returns:
            Tensor[np.floating]: The output of the neural network after the forward pass.
        """
        if len(inputs) != self.weights[0].shape[1]:
            raise ValueError(
                f"Input size {len(inputs)} does not match expected size {self.weights[0].shape[1]}"
            )

        return self._forward_pass(inputs)[-1]

    def _forward_pass(
        self,
        inputs: Tensor[np.floating],
        training: bool = False,
    ) -> list[Tensor[np.floating]]:
        """
        Perform a forward pass through the neural network.
        Args:
            inputs (Tensor[np.floating]): The input data for the neural network.
            training (bool): Flag indicating if its training or not
        Returns:
            list[Tensor[np.floating]]: The output of each layer of the neural network.
        """
        layer_outputs = [inputs.reshape((-1, 1))]

        for i in range(self.num_hidden_layers + 1):
            z = (self.weights[i] @ layer_outputs[-1]) + self.biases[i]

            if i != self.num_hidden_layers:
                activation = self.hidden_activation(z)
                if self.dropout and training:
                    self.dropout(activation)
            else:
                activation = self.output_activation(z)

            layer_outputs.append(activation)

        return layer_outputs

    @staticmethod
    def _set_requires_grad(
        data: list[tuple[Tensor | ArrayLike, str]],
    ) -> list[tuple[Tensor, str]]:
        for i, item in enumerate(data):
            if isinstance(item[0], Tensor):
                item[0].requires_grad = True
            else:
                data[i] = (Tensor(item[0], requires_grad=True), item[1])

        return data  # type: ignore

    def train(
        self,
        data_train: list[tuple[Tensor | ArrayLike, str]],
        data_evaluate: list[tuple[Tensor | ArrayLike, str]],
        config: TrainingConfig = TrainingConfig(),
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
        # Set requires_grad to True for all input data
        d_train = self._set_requires_grad(data_train)
        d_evaluate = self._set_requires_grad(data_evaluate)

        for w, b in zip(self.weights, self.biases):
            w.requires_grad = True
            b.requires_grad = True

        last_epoch_loss = float("inf")
        patience_counter = 0

        metrics = {"losses": [], "train_acc": [], "test_acc": []}

        best_weights = []
        best_biases = []
        best_accuracy = 0.0

        for epoch in range(config.epochs + 1):
            loss = self._backward(config.lr.learning_rate, d_train)
            current_epoch_loss = np.mean(loss)

            val_accuracy = self.evaluate(d_evaluate)

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
                train_accuracy = self.evaluate(d_train)
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
            kwds = {f"{c.WEIGHT_PREFIX}{i}": w for i, w in enumerate(self.weights)}
            kwds.update({f"{c.BIAS_PREFIX}{i}": b for i, b in enumerate(self.biases)})

            np.savez(self.params_file, **kwds)

        if config.debug:
            self._plot_metrics_train(**metrics)

        for w, b in zip(self.weights, self.biases):
            w.requires_grad = False
            b.requires_grad = False

    def _backward(
        self, lr: float, data: list[tuple[Tensor, str]]
    ) -> list[Tensor[np.floating]]:
        """
        Perform the backward pass of the neural network, updating weights and biases.
        Args:
            data (list[tuple[Tensor, str]]): A list of tuples where each tuple contains
                an input array and the corresponding expected label.
        Returns:
            list[np.floating]: The losses of the training.
        """
        data_array = np.array(data, dtype=object)
        n_batches = (len(data_array) + self.batch_size - 1) // self.batch_size
        batches = np.array_split(data_array, n_batches)

        losses: list[Tensor] = []

        for data_batch in batches:
            weights_batch_gradient = [op.zeros_like(w) for w in self.weights]
            biases_batch_gradient = [op.zeros_like(b) for b in self.biases]

            for inputs, expected_label in data_batch:
                expected = self.encoder(expected_label)
                outputs_layers = self._forward_pass(inputs, training=True)
                predicted = outputs_layers[-1]

                loss = None
                if isinstance(self.loss, CategoricalCrossentropy):
                    # This approach is used because the current autograd implementation does not support
                    # the computation of the Jacobian matrix, which is necessary for calculating the gradient.
                    # TODO: Implement the Jacobian matrix in the autograd system and remove this workaround.
                    # This approach is functional until the Jacobian matrix support is added.
                    loss = op.cce(outputs_layers[-1], expected)
                else:
                    loss = self.loss(expected, predicted)

                losses.append(loss)

                loss.backward()

                for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                    weights_batch_gradient[i] += w.grad
                    biases_batch_gradient[i] += b.grad

                    w.clear_grad()
                    b.clear_grad()

            batch_size = len(data_batch)

            for i, (w, b) in enumerate(
                zip(weights_batch_gradient, biases_batch_gradient)
            ):
                weight_norm = np.linalg.norm(w)
                if weight_norm > self.MAX_DELTA_NORM:
                    w *= self.MAX_DELTA_NORM / (weight_norm + c.EPSILON)

                weights_batch_gradient[i] = w / batch_size
                biases_batch_gradient[i] = b / batch_size

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
            self.encoder.decode(self.forward_pass(input)) == label
            for input, label in data
        )
        return correct / len(data)

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
