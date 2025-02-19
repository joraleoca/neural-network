import cupy as cp

from .tensor.device import Device


class _ConfigMeta(type):
    _default_device = "auto"

    @property
    def default_device(cls) -> str:
        if cls._default_device == Device.AUTO:
            try:
                cuda = cp.cuda.is_available()
            except Exception:
                cuda = False

            cls._default_device = Device.CUDA if cuda else Device.CPU

        return cls._default_device

    @classmethod
    def set_default_device(cls, device: str | Device) -> str:
        """
        Set the default device for the neural network.

        Args:
            device: Device to use. Can be "cpu", "cuda" or "auto".
        """
        if device not in Device:
            raise ValueError("device not in the Device options. Got ")

        cls._default_device = device.value if isinstance(device, Device) else device


class Config(metaclass=_ConfigMeta):
    """Configuration class for the neural network."""
