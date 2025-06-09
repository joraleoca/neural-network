from abc import ABC, abstractmethod

import numpy as np

from src.tensor import Tensor
from .parameter import Parameter


class Layer(ABC):
    """Abstract class for a layer in a neural network."""

    @abstractmethod
    def __call__(
        self,
        *args: Tensor[np.floating],
    ) -> Tensor[np.floating]:
        """
        Forward pass of the layer.
        Args:
            args (*Tensor): The input data to the layer.
        Returns:
            Tensor: The output of the layer.
        """
        pass

    @property
    def parameters(self) -> list[Parameter]:
        """
        Returns the parameters of the layer.
        Returns:
            list[Parameter]: The parameters of the layer.
        """
        params = set()

        if self.__slots__:
            iter_ = self.__slots__
            if isinstance(iter_, str):
                iter_ = [iter_]
        else:
            iter_ = self.__dict__.keys()

        for name in iter_:
            if (attr := getattr(self, name, None)) is not None:
                _add_params(params, attr)

        return list(params)


def _add_params(params: set[Parameter], attr) -> None:
    """
    Helper function to add parameters to a set.
    Args:
        params (set[Parameter]): The set of parameters.
        attr: The attribute to add.
    """
    if isinstance(attr, Parameter):
        params.add(attr)
    elif hasattr(attr, "parameters"):
        params.update(attr.parameters)
    elif isinstance(attr, dict):
        params.update(p.parameters for p in attr.values() if hasattr(p, "parameters"))
    elif isinstance(attr, (list, tuple, set)):
        for item in attr:
            _add_params(params, item)
