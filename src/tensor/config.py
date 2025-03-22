from abc import ABCMeta

import cupy as cp
import numpy as np
from numpy.typing import DTypeLike

from .device import Device


class _ConfigMeta(ABCMeta):
    _default_device: str = "auto"
    _default_dtype: np.dtype = np.dtype(np.float32)
    _grad: bool = True

    @property
    def default_device(cls) -> str:
        if cls._default_device == Device.AUTO:
            try:
                cuda = cp.cuda.is_available()
            except Exception:
                cuda = False

            cls._default_device = Device.CUDA.value if cuda else Device.CPU.value

        return cls._default_device

    @classmethod
    def set_default_device(cls, device: str | Device) -> None:
        """
        Set the default device for the neural network.

        Args:
            device: Device to use. Can be "cpu", "cuda" or "auto".
        """
        if isinstance(device, Device):
            device = device.value

        if device not in Device:
            raise ValueError(f"device not in the Device options. Got {device}")

        cls._default_device = device

    @property
    def default_dtype(cls) -> np.dtype:
        return cls._default_dtype

    @classmethod
    def set_default_dtype(cls, dtype: np.dtype | cp.dtype | DTypeLike) -> None:
        """
        Set the default data type for the neural network.

        Args:
            dtype: Data type to use.
        """
        cls._default_dtype = np.dtype(dtype)

    @property
    def grad(cls) -> bool:
        return cls._grad

    @classmethod
    def set_grad(cls, grad: bool) -> None:
        """
        Set the gradient computation for the neural network.

        Args:
            grad: Whether to compute the gradients.
        """
        cls._grad = grad

    @property
    def default_module(cls):
        match cls.default_device:
            case Device.CUDA:
                return cp
            case Device.CPU:
                return np
            case _:
                raise ValueError(f"Invalid default device in Tensor configuration. Got {cls.default_device}")
