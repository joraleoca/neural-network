import numpy as np
from numpy.typing import NDArray

from typing import Final, Any

from activation import FunctionActivation
from initialization import Initializator, HeUniform
from optimizer import Optimizer, SGD


class NeuralNetwork:
    __slots__ = (
        "weights",
        "biases",
        "num_hidden_layers",
        "classes",
        "hidden_activation",
        "output_activation",
        "optimizer",
        "learning_rate",
        "momentum",
        "_label_to_index",
        "_last_weights_updates",
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

    learning_rate: float
    momentum: float

    EPSILON: Final[float] = 1e-15
    MAX_DELTA_NORM: Final[float] = float("inf")

    def __init__(
        self,
        network_structure: list[int],
        classes: tuple[str, ...],
        hidden_activation: FunctionActivation,
        output_activation: FunctionActivation,
        *,
        learning_rate: float = 0.001,
        initializator: Initializator = HeUniform(),
        optimizer: Optimizer = SGD(),
        random_seed=None,
    ) -> None:
        """
        Initializes the neural network with the given structure, activation functions, and optional random seed.
        Args:
            network_structure (list[int]): A list representing the number of nodes in each layer of the network.
            classes (tuple[str, ...]): A tuple of class labels for the output layer.
            hidden_activation (FunctionActivation): The activation function to use for hidden layers.
            output_activation (FunctionActivation): The activation function to use for the output layer.
            initializator (Initializator): The weights and biases initializator
            random_seed (optional): An optional seed for random number generation. Defaults to None.
        Raises:
            ValueError: If the number of output nodes does not match the number of output classes.
        """
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if not network_structure or any(n <= 0 for n in network_structure):
            raise ValueError("Invalid network structure")
        if network_structure[-1] != len(classes):
            raise ValueError(
                "The network must have the same output nodes as output classes"
            )

        self.learning_rate = learning_rate
        self.num_hidden_layers = len(network_structure) - 2  # First and last layer
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        if self.optimizer:
            self.optimizer.learning_rate = learning_rate
        self.classes = classes
        self._label_to_index = {c: i for i, c in enumerate(classes)}

        self.weights = initializator.initializate(
            network_structure, rng=np.random.default_rng(random_seed)
        )
        self.biases = [
            np.zeros((network_structure[i + 1], 1))
            for i in range(len(network_structure) - 1)
        ]

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
        output, _ = self._forward_pass(inputs)
        return output[-1]

    def _forward_pass(
        self, inputs: NDArray[np.floating[Any]]
    ) -> tuple[list[NDArray[np.floating[Any]]], list[NDArray[np.floating[Any]]]]:
        """
        Perform a forward pass through the neural network.
        Args:
            inputs (NDArray[np.floating[Any]]): The input data for the neural network.
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
        self, data: list[tuple[np.ndarray, str]], losses: list[np.floating]
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
            outputs_layers, inputs_layers = self._forward_pass(inputs)
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

            # Update the parameters based on the gradient
            self.optimizer.optimize(
                gradient,
                weights=self.weights,
                biases=self.biases,
                output_layers=outputs_layers,
            )

    def train(
        self,
        data_train: list[tuple[np.ndarray, str]],
        data_evaluate: list[tuple[np.ndarray, str]],
        *,
        epochs: int = 1000,
        debug: bool = False,
    ) -> None:
        losses = []

        for epoch in range(epochs + 1):
            self._backwards(data_train, losses)

            if debug:
                accuracy = self.evaluate(data_evaluate)
                current_epoch_loss = np.mean(
                    losses[epoch * len(data_train) : (epoch + 1) * len(data_train)]
                )
                print(
                    f"Epoch {epoch}, Accuracy: {accuracy:.4f}, Loss: {current_epoch_loss:.4f}"
                )

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
