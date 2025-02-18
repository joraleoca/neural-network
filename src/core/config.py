class Config:
    """Configuration class for the neural network."""
    default_device = "auto"

    @classmethod
    def set_default_device(cls, device: str):
        """
        Set the default device for the neural network.

        Args:
            device: Device to use. Can be "cpu", "gpu" or "auto".
        """
        cls.default_device = device
