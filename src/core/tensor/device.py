from enum import StrEnum


class Device(StrEnum):
    """Class with constants representing the available devices for tensor creation."""

    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"
