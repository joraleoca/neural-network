from .trainable import Trainable
from src.tensor import Tensor, T, op
from src.constants import EPSILON


class BatchNorm(Trainable):
    """BatchNorm layer."""

    __slots__ = "mean", "var", "weights", "biases", "momentum", "rng"

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

        self.mean = op.zeros(shape)
        self.var = op.ones(shape)
        self.weights = op.ones(shape, requires_grad=self.requires_grad)
        self.biases = op.zeros(shape, requires_grad=self.requires_grad)
        self.momentum = momentum

    def __call__(self, data: Tensor[T]) -> Tensor[T]:
        if not Tensor.training:
            data = (data - self.mean) / op.sqrt(self.var + EPSILON)
        else:
            match data.ndim:
                case 4:
                    axis = (0, 2, 3)
                case 3:
                    axis = (0, 1)
                case 2:
                    axis = 0
                case _:
                    raise ValueError(f"Data ndim must be in (2, 3, 4). Got {data.ndim}")

            mean = data.mean(axis, keepdims=True)
            var = ((data - mean) ** 2).mean(axis, keepdims=True)

            data = (data - mean) / op.sqrt(var + EPSILON)

            # Update the moving average of mean and variance without gradient tracking
            with Tensor.no_grad():
                self.mean = self.mean * (1 - self.momentum) + mean * self.momentum
                self.var = self.var * (1 - self.momentum) + var * self.momentum

        return self.weights * data + self.biases
