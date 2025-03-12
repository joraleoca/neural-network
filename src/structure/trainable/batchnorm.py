from typing import Any, ClassVar

from .trainable import Trainable
from src.tensor import Tensor, T, op


class BatchNorm(Trainable):
    """BatchNorm layer."""

    __slots__ = "mean", "var", "weights", "biases", "momentum", "rng"

    required_fields: ClassVar[tuple[str, ...]] = ("p",)

    def __init__(self, num_features: int, num_dims: int, momentum: float = 0.1) -> None:
        """
        Initializes a new BatchNorm layer in the neural network.

        Args:
            num_features (int): The number of features in the input tensor.
            num_dims (int): The number of dimensions in the input tensor.
            momentum (float): The momentum for the moving average of the mean and variance.
            rng (Any): A random number generator instance for initializing weights.
        """
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)

        self.mean = op.zeros(shape, requires_grad=self.requires_grad)
        self.var = op.ones(shape, requires_grad=self.requires_grad)
        self.weights = op.ones(shape, requires_grad=self.requires_grad)
        self.biases = op.zeros(shape, requires_grad=self.requires_grad)
        self.momentum = momentum

    def __call__(self, data: Tensor[T]) -> Tensor[T]:
        if not data.requires_grad:
            data = data - self.mean / op.sqrt(self.var)
        else:
            assert len(data.shape) in (2, 4), "BatchNorm is only supported for Dense and Convolutional layers."

            if len(data.shape) == 2:
                mean = data.mean(axis=0)
                var = ((data - mean) ** 2).mean(axis=0)
            else:
                mean = data.mean(axis=(0, 2, 3), keepdims=True)
                var = ((data - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)

            data = (data - mean) / op.sqrt(var)

            self.mean *= 1 - self.momentum
            self.mean += mean * self.momentum

            self.var *= 1 - self.momentum
            self.var += var * self.momentum

        return self.weights * data + self.biases

    @property
    def input_dim(self) -> int:
        return self.mean.shape[1]

    @property
    def output_dim(self) -> int:
        return self.mean.shape[1]

    def data_to_store(self) -> dict[str, Any]:
        return {
            "mean": self.mean.data,
            "var": self.var.data,
            "weights": self.weights.data,
            "biases": self.biases.data,
            "momentum": self.momentum,
        }

    @staticmethod
    def from_data(data: dict[str, Any]) -> "BatchNorm":
        out = BatchNorm.__new__(BatchNorm)
        out.mean = Tensor(data["mean"])
        out.var = Tensor(data["var"])
        out.weights = Tensor(data["weights"])
        out.biases = Tensor(data["biases"])
        out.momentum = Tensor(data["momentum"])
        out.requires_grad = True
        return out
