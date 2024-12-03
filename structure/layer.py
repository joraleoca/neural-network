from typing import Any

import numpy as np
from numpy.random import Generator

from core import Tensor, op
from activation import ActivationFunction
from initialization import Initializator
from regularization import dropout


class Layer:
    """
    Represents a layer in a neural network.
    """

    __slots__ = [
        "in_features",
        "out_features",
        "activation_function",
        "weights",
        "biases",
        "forward_output",
        "rng",
        "_requires_grad",
        "_weights_grad",
        "_biases_grad",
        "_times_grad",
        "_initializer",
    ]

    in_features: int
    out_features: int

    activation_function: ActivationFunction | None

    weights: Tensor[np.floating]
    biases: Tensor[np.floating]

    forward_output: Tensor[np.floating]

    rng: Any

    _weights_grad: Tensor[np.floating]
    _biases_grad: Tensor[np.floating]

    _times_grad: int

    _requires_grad: bool

    # Store only to use the first time forward is called, after that deleted
    _initializer: Initializator | None

    def __init__(
        self,
        features: int | tuple[int, int],
        activation_function: ActivationFunction | None = None,
        weights_initializer: Initializator | None = None,
        *,
        rng: Generator | None = None,
    ):
        """
        Initializes a new layer in the neural network.
        Args:
            features (int | tuple[int, int]):
                If int, the number of nodes in the layer, the in features are inferred from the first call to forward.
                If tuple, the number of (in features, out features)
            activation_function (ActivationFunction | None): The activation function to be used by this layer.
            weights_initializer (Initializator | None): The initializer for the weights of this layer. If None, the weights are not initialized.
            rng (Generator | None): A random number generator instance for initializing weights.
        Raises:
            ValueError: If any features is incorrect.
        """
        if isinstance(features, int):
            if features <= 0:
                raise ValueError(
                    f"The layer must have positive out features. Got {features}"
                )

            self.out_features = features
            self._initializer = weights_initializer
        elif isinstance(features, tuple):
            if len(features) != 2:
                raise ValueError(
                    f"The features must have len in and out features only. Got {features}."
                )
            for f in features:
                if f <= 0:
                    raise ValueError(
                        f"The layer must have positive features. Got {features}"
                    )

            self.in_features, self.out_features = features
            self._initializer = weights_initializer
            self._initialize_weights(rng)

        self.activation_function = activation_function
        self.biases = op.zeros((self.out_features, 1))
        self._requires_grad = False
        self.rng = rng

        self._weights_grad = None
        self._biases_grad = None
        self._times_grad = 0

    def forward(
        self,
        data: Tensor[np.floating],
        *,
        dropout_rate: float = 0.0,
    ) -> Tensor[np.floating]:
        """
        Forward pass of the layer.

        Args:
            data (Tensor): The input data to the layer.
            dropout_rate (float): The dropout rate for the layer.

        Returns:
            Tensor: The output of the layer.
        """
        if not hasattr(self, "in_features"):
            self.in_features = data.shape[0]
            self._initialize_weights(self.rng)

        if data.shape[0] != self.in_features:
            raise ValueError(
                f"Data has {data.shape[1]} features but the layer has {self.in_features} features."
            )

        self.forward_output = (self.weights @ data) + self.biases

        if self.activation_function:
            self.forward_output = self.activation_function(self.forward_output)

        self.forward_output = dropout(self.forward_output, p=dropout_rate, rng=self.rng)

        return self.forward_output

    def backward(self) -> None:
        """
        Accumulates the gradients of the layer parameters.\n
        The gradients of these are cleared.

        Args:
            batch_size (int): The size of the batch used in the forward pass.
        """
        assert hasattr(
            self, "forward_output"
        ), "The forward method must be called first."

        if not self._weights_grad or not self._biases_grad:
            self._weights_grad = Tensor(self.weights.grad, requires_grad=False)
            self._biases_grad = Tensor(self.biases.grad, requires_grad=False)
            self._times_grad = 1
        else:
            self._weights_grad += self.weights.grad
            self._biases_grad += self.biases.grad
            self._times_grad += 1

        self.weights.clear_grad()
        self.biases.clear_grad()

    def clear_grad(self) -> None:
        """Clears the gradient of the layer or initializes it if it does not exist."""
        self.weights.clear_grad()
        self.biases.clear_grad()
        self._weights_grad = None
        self._biases_grad = None

    @property
    def requires_grad(self) -> bool:
        """Returns whether the layer requires gradients."""
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool) -> None:
        """Sets whether the layer requires gradients."""
        self._requires_grad = requires_grad
        if hasattr(self, "weights"):
            self.weights.requires_grad = requires_grad
        self.biases.requires_grad = requires_grad

        if not requires_grad:
            self.clear_grad()

    @property
    def weights_grad(self) -> Tensor[np.floating]:
        """Returns the accumulated gradients of the weights."""
        return self._weights_grad / self._times_grad

    @property
    def biases_grad(self) -> Tensor[np.floating]:
        """Returns the accumulated gradients of the biases."""
        return self._biases_grad / self._times_grad

    # Methods for lazy feed-forward network
    @property
    def initializer(self) -> Initializator | None:
        """Returns the initializer of the layer."""
        return self._initializer

    @initializer.setter
    def initializer(self, initializer: Initializator) -> None:
        """Sets the initializer of the layer."""
        self._initializer = initializer

    def _initialize_weights(self, rng: Generator | None = None) -> None:
        """Initializes the weights of the layer."""
        if self._initializer:
            self.weights = self._initializer.initialize(
                (self.out_features, self.in_features), rng=rng
            )
            self.weights.requires_grad = self._requires_grad
        self._initializer = None
