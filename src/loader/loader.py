from pathlib import Path
import os

import numpy as np
from numpy.typing import NDArray

from src.structure import Layer, layer_from_name
from src import constants as c


class Loader:
    """
    Loader is a class responsible for loading and validating neural network parameters from a file path.
    """

    __slots__ = "path"

    path: Path

    def __init__(self, path: str | os.PathLike = c.FILE_NAME):
        """
        Initializes the loader with the given file path.

        Args:
            path (str | os.PathLike): The path to the file to be loaded.

        Raises:
            FileNotFound: If the file is not found.
            LoadError: If the file has any error
        """
        self.path = Path(path)
        self._validate_file()

    def load(
        self,
    ) -> list[Layer]:
        """
        Load neural network parameters in-place from a specified file path.

        Returns:
            list[Layer]: List neural network layers.
        Raises: LoadError
        """
        params = self._load_npz_file()

        structure = params[c.STRUCTURE_STR]
        network_structure = []

        for layer in structure:
            name, number = layer.split(" ")

            layer_type = layer_from_name(name)

            data = {field: params[f"{field} {number}"] for field in layer_type.required_fields}

            network_structure.append(layer_type.from_data(data))

        return network_structure

    def _validate_file(self) -> None:
        """
        Validates the file path.

        Raises:
            FileNotFoundError: If the file does not exist at the given path.
            LoadError: If the file extension is not .npz or is not readable.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"File {self.path} not found.")
        if self.path.suffix != ".npz":
            raise LoadError(
                f"Path must be a numpy .npz file. Got {self.path.suffix}"
            )
        if not os.access(self.path, os.R_OK):
            raise LoadError(f"File {self.path} is not readable.")

    def _load_npz_file(self) -> dict[str, NDArray]:
        """
        Loads parameters from a .npz file.

        Returns:
            dict[str, NDArray]: A dictionary where the keys are strings and the values are numpy arrays.

        Raises:
            LoadError: If there is an error loading the parameters or if the parameters file is empty.
        """
        try:
            with np.load(self.path) as p:
                params = {key: p[key] for key in p}
        except Exception as e:
            raise LoadError(f"Error loading parameters: {e}")

        if not params:
            raise LoadError("Parameters file is empty")
        return params


class LoadError(Exception):
    """
    Exception raised for errors in loading parameters.
    """

    pass
