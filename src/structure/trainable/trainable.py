from abc import ABC
from typing import Any

from ..layer import Layer


class Trainable(Layer, ABC):
    __slots__ = "rng", "initializer", "_requires_grad"

    rng: Any

    _requires_grad: bool

    def __init__(self, *, requires_grad: bool = True, rng: Any = None) -> None:
        self._requires_grad = requires_grad
        self.rng = rng

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool) -> None:
        self._requires_grad = requires_grad

        for param in self.parameters:
            param.requires_grad = requires_grad
