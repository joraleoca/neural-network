import numpy as np
from numpy.typing import NDArray

from typing import Final, Any
from copy import deepcopy

from activation import FunctionActivation
from optimizer import Optimizer
from config import NeuralNetworkConfig, TrainingConfig
from regularization import Dropout

FILE_NAME = "params"


class NeuralNetwork:
    __slots__ = (
        "weights",
        "biases",
        "num_hidden_layers",
        "classes",
        "hidden_activation",
        "output_activation",
        "optimizer",
        "dropout",
        "_label_to_index",
    )

    weights: list[NDArray[np.floating[Any]]]
    biases: list[NDArray[np.floating[Any]]]

    num_hidden_layers: Final[int]

    # Set of possible output classes
    classes: tuple[str, ...]
    _label_to_index: dict[str, int]

    # Function activators
    hidden_activation: FunctionActivation
    output_activation: FunctionActivation

    optimizer: Optimizer
    dropout: Dropout | None

    EPSILON: Final[float] = 1e-15
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
        self.optimizer = config.optimizer
        self.dropout = config.dropout
        self.classes = config.classes
        self._label_to_index = {c: i for i, c in enumerate(config.classes)}

        if not config.load_parameters:
            self.weights = config.initializator.initializate(
                config.network_structure, rng=np.random.default_rng(config.random_seed)
            )
            self.biases = [
                np.zeros((config.network_structure[i + 1], 1))
                for i in range(len(config.network_structure) - 1)
            ]
        else:
            try:
                with np.load(config.load_file_name, "r") as params:
                    self.weights = [
                        params[f"weights{i}"]
                        for i in range(len(config.network_structure) - 1)
                    ]
                    self.biases = [
                        params[f"biases{i}"]
                        for i in range(len(config.network_structure) - 1)
                    ]
            except Exception as e:
                print(f"Error loading parameters: {e}")
                return

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

    def one_hot_encode(self, label: str) -> NDArray[np.floating[Any]]:
        """
        Converts a label into a one-hot encoded vector.
        Args:
            label (str): The label to be one-hot encoded.
        Returns:
            NDArray[np.floating[Any]]: A one-hot encoded vector representing the label.
        Raises:
            ValueError: If the label is not found in the classes.
        """
        if label not in self._label_to_index:
            raise ValueError("Label is not in classes")

        encode = np.zeros((len(self.classes), 1), dtype=np.float64)
        encode[self._label_to_index[label]] = 1

        return encode

    def _backwards(
        self, lr: float, data: list[tuple[np.ndarray, str]], losses: list[np.floating]
    ) -> None:
        """
        Perform the backward pass of the neural network, updating weights and biases.
        Args:
            data (list[tuple[np.ndarray, str]]): A list of tuples where each tuple contains
                an input array and the corresponding expected label.
            losses (list[np.floating]): A list to store the loss values for each input, in place.
        Returns:
            None: This method modifies the weights and biases in place.
        """
        for inputs, expected_label in data:
            expected = self.one_hot_encode(expected_label)
            outputs_layers, inputs_layers = self._forward_pass(inputs, training=True)
            predicted = outputs_layers[-1]

            losses.append(self.categorical_crossentropy(expected, predicted))

            gradient = [self.categorical_crossentropy_gradient(expected, predicted)]

            for i in reversed(range(len(self.weights) - 1)):
                gradient.append(
                    (self.weights[i + 1].T @ gradient[-1])
                    * self.hidden_activation.derivative(inputs_layers[i])
                )

            # Normalize gradient by dividing by its norm (L2 norm)
            for i, d in enumerate(gradient):
                norm = np.linalg.norm(d)
                if norm > self.MAX_DELTA_NORM:
                    gradient[i] = np.multiply(d, self.MAX_DELTA_NORM / norm)

            gradient.reverse()

            weight_gradient = [
                (g @ outputs_layers[i].T) for i, g in enumerate(gradient)
            ]
            biases_gradient = list(gradient)

            # Update the parameters based on the gradient
            self.optimizer.optimize_weights(
                lr,
                weight_gradient,
                weights=self.weights,
            )
            self.optimizer.optimize_biases(
                lr,
                biases_gradient,
                biases=self.biases,
            )

    def train(
        self,
        data_train: list[tuple[np.ndarray, str]],
        data_evaluate: list[tuple[np.ndarray, str]],
        config: TrainingConfig = TrainingConfig(),
    ) -> None:
        losses = []
        last_epoch_loss = float("inf")
        patience_counter = 0

        best_weights = []
        best_biases = []
        best_accuracy = 0.0

        for epoch in range(config.epochs + 1):
            self._backwards(config.lr.learning_rate, data_train, losses)

            current_epoch_loss = np.mean(
                losses[epoch * len(data_train) : (epoch + 1) * len(data_train)]
            )

            val_accuracy = self.evaluate(data_evaluate)

            if val_accuracy > best_accuracy:  # Change criterion to validation accuracy
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

            if config.lr.patience_update == 0 or epoch % config.lr.patience_update == 0:
                config.lr.update()

            if config.debug:
                train_accuracy = self.evaluate(data_train)
                print(
                    f"Epoch {epoch}, "
                    f"Accuracy: {val_accuracy:.4f}, "
                    f"Loss: {current_epoch_loss:.4f}, "
                    f"Accuracy_train: {train_accuracy:.4f}"
                )

        self.weights = best_weights
        self.biases = best_biases

        if config.store:
            kwds = {f"weights{i}": w for i, w in enumerate(self.weights)}
            kwds.update({f"biases{i}": b for i, b in enumerate(self.biases)})

            np.savez(FILE_NAME, **kwds)

    def categorical_crossentropy(
        self,
        expected: NDArray[np.floating[Any]],
        predicted: NDArray[np.floating[Any]],
    ) -> np.floating[Any]:
        """
        Compute the categorical cross-entropy loss function with L2 regularization (weight decay).
        """
        if expected.shape != predicted.shape:
            raise ValueError(
                f"Shape mismatch: expected {expected.shape}, got {predicted.shape}"
            )

        clipped_predicted = np.clip(predicted, self.EPSILON, 1 - self.EPSILON)

        # Apply label smoothing to the expected output
        smoothing_factor = 0.1
        smoothed_expected = expected * (1 - smoothing_factor) + (
            smoothing_factor / len(self.classes)
        )

        # Calculate cross-entropy loss
        ce_loss = -np.sum(smoothed_expected * np.log(clipped_predicted))

        return ce_loss

    def categorical_crossentropy_gradient(
        self,
        expected: NDArray[np.floating[Any]],
        predicted: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """
        Compute the gradient of the categorical cross-entropy loss function.
        The weight decay gradient will be handled in the optimizer.
        """
        if expected.shape != predicted.shape:
            raise ValueError(
                f"Shape mismatch: expected {expected.shape}, got {predicted.shape}"
            )

        clipped_predicted = np.clip(predicted, self.EPSILON, 1 - self.EPSILON)

        # Apply label smoothing to the expected output
        smoothing_factor = 0.1
        smoothed_expected = expected * (1 - smoothing_factor) + (
            smoothing_factor / len(self.classes)
        )

        return clipped_predicted - smoothed_expected

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
        correct = sum(
            self.classes[np.argmax(self.forward_pass(input))] == label
            for input, label in data
        )
        return correct / len(data)
