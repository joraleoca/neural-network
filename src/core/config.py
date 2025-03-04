import cupy as cp
import numpy as np
from numpy.typing import DTypeLike

from .tensor.device import Device


class _ConfigMeta(type):
    _default_device: str = "auto"
    _default_dtype: np.dtype = np.dtype(np.float32)

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

class Config(metaclass=_ConfigMeta):
    """Configuration class for default parameters of the tensors."""
