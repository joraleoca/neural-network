from pathlib import Path
import os
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .exceptions import ParameterLoadError
from core.constants import FILE_NAME, WEIGHT_PREFIX, BIAS_PREFIX


class ParameterLoader:
    """
    ParameterLoader is a class responsible for loading and validating neural network parameters from a file path.

    Attributes:
        path (Path): The path to the file containing the neural network parameters.
    """

    path: Path

    def __init__(self, path: str | os.PathLike = FILE_NAME):
        """
        Initializes the loader with the given file path.

        Args:
            path (str | os.PathLike): The path to the file to be loaded.

        Raises:
            FileNotFound: If the file is not found.
            ParameterLoadError: If the file has any error
        """
        self.path = Path(path)
        self._validate_file()

    def load(
        self,
    ) -> tuple[list[NDArray[np.floating[Any]]], list[NDArray[np.floating[Any]]], int]:
        """
        Load neural network parameters in-place from a specified file path.

        Returns:
            tuple[list[NDArray[np.floating[Any]]], list[NDArray[np.floating[Any]]], int]:
                [weights, biases, num of hidden layers]

        Raises:
            ParameterLoadError:
            If the file does not exist, is not a .npz file, contains invalid data,
            or if there is a mismatch in the expected number of layers or parameter shapes.
        """
        params = self._load_npz_file()
        num_hidden_layers = self._count_layers(params)
        weights, biases = self._create_weights_biases(params)
        return weights, biases, num_hidden_layers

    def _validate_file(self) -> None:
        """
        Validates the file path.

        Raises:
            FileNotFoundError: If the file does not exist at the given path.
            ParameterLoadError: If the file extension is not .npz or is not readable.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"File {self.path} not found.")
        if self.path.suffix != ".npz":
            raise ParameterLoadError(
                f"Path must be a numpy .npz file. Got {self.path.suffix}"
            )
        if not os.access(self.path, os.R_OK):
            raise ParameterLoadError(f"File {self.path} is not readable.")

    def _load_npz_file(self) -> dict[str, NDArray]:
        """
        Loads parameters from a .npz file.

        Returns:
            dict[str, NDArray]: A dictionary where the keys are strings and the values are numpy arrays.

        Raises:
            ParameterLoadError: If there is an error loading the parameters or if the parameters file is empty.
        """
        try:
            with np.load(self.path) as p:
                params = {key: p[key] for key in p}
        except Exception as e:
            raise ParameterLoadError(f"Error loading parameters: {e}")

        if not params:
            raise ParameterLoadError("Parameters file is empty")
        return params

    def _count_layers(self, params: dict[str, NDArray]) -> int:
        """
        Validates the parameters dictionary to ensure it contains an even number of keys,
        representing pairs of weights and biases for each layer in a neural network.

        Args:
            params (dict[str, NDArray]): A dictionary where keys are parameter names and values are numpy arrays.

        Returns:
            int: The number of layers in the neural network minus one.

        Raises:
            ParameterLoadError: If the number of parameters is not even, indicating a mismatch in weights/biases pairs.
        """
        num_layers_params = len(params.keys()) // 2

        if len(params.keys()) % 2 != 0:
            raise ParameterLoadError(
                "Invalid parameter file: Expected even number of parameters (weights/biases pairs)"
            )

        return num_layers_params - 1

    def _create_weights_biases(
        self, params: dict[str, NDArray]
    ) -> tuple[list[NDArray[np.floating[Any]]], list[NDArray[np.floating[Any]]]]:
        """
        Creates and validates weights and biases for each layer from the given parameters.

        Args:
            params (dict[str, NDArray]): A dictionary containing weights and biases for each layer.
            The keys should be in the format "weights{i}" and "biases{i}" where {i} is the layer index.

        Returns:
            tuple[list[NDArray[np.floating[Any]]], list[NDArray[np.floating[Any]]]]:
            A tuple containing two lists: [weights, biases]
        """
        weights = []
        biases = []

        prev_output_size: int | None = None
        for i in range(len(params) // 2):
            weight = params[f"{WEIGHT_PREFIX}{i}"]
            bias = params[f"{BIAS_PREFIX}{i}"]

            self._validate_layer_params(i, weight, bias, prev_output_size)

            prev_output_size = weight.shape[0]
            weights.append(weight)
            biases.append(bias)

        return weights, biases

    def _validate_layer_params(
        self,
        layer_index: int,
        weight: NDArray,
        bias: NDArray,
        prev_output_size: int | None,
    ) -> None:
        """
        Validates the parameters of a neural network layer.
        Args:
            layer_index (int): The index of the layer being validated.
            weight (NDArray): The weight matrix of the layer.
            bias (NDArray): The bias vector of the layer.
            prev_output_size (int | None): The output size of the previous layer, or None if this is the first layer.
        Raises:
            ParameterLoadError: If any of the following conditions are met:
                - The weight or bias contains NaN values.
                - The weight or bias contains infinite values.
                - The weight is not a 2D array.
                - The bias is not a 1D array.
                - The dimensions of the weight and bias do not match.
                - The input dimension of the weight does not match the output dimension of the previous layer.
        """
        if np.any(np.isnan(weight)) or np.any(np.isnan(bias)):
            raise ParameterLoadError(
                f"Layer {layer_index}: Parameters contain NaN values"
            )
        if np.any(np.isinf(weight)) or np.any(np.isinf(bias)):
            raise ParameterLoadError(
                f"Layer {layer_index}: Parameters contain infinite values"
            )

        if len(weight.shape) != 2:
            raise ParameterLoadError(
                f"Weights must be 2D arrays. Got {len(weight.shape)} in layer {layer_index}"
            )
        if bias.shape[1] != 1:
            raise ParameterLoadError(
                f"Biases must have 1 number per node. Got {bias.shape[0]}"
            )

        if weight.shape[0] != bias.shape[0]:
            raise ParameterLoadError(
                f"Layer {layer_index}: Weight and bias dimensions mismatch. Weights: {weight.shape}, Bias: {bias.shape}"
            )

        if prev_output_size is not None and weight.shape[1] != prev_output_size:
            raise ParameterLoadError(
                f"Layer {layer_index}: Input dimension ({weight.shape[1]}) doesn't match previous layer output dimension ({prev_output_size})"
            )
