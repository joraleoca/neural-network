import cupy as cp
import numpy as np
import pytest

from src.core import Tensor
from src.core.tensor import op
from .utils import assert_data, assert_grad


@pytest.fixture
def sample_2d_tensor():
    """Fixture for a standard 2D tensor used across multiple tests."""
    return Tensor([[1, 2, 3], [4, 5, 6]], dtype=np.float32)


class TestTensorCreation:
    def test_basic_creation(self):
        """Test basic tensor creation with default parameters."""
        data = [1, 2, 3]
        tensor = Tensor(data, dtype=np.float32)
        assert_data(tensor, np.array(data, dtype=cp.float32))
        assert not tensor.requires_grad
        assert tensor.device == ("cuda" if cp.cuda.is_available() else "cpu")

    @pytest.mark.parametrize(
        "device",
        [
            "cuda",
            "cpu",
        ],
    )
    def test_cpu_creation(self, device):
        """Test tensor creation on devices."""
        if device == "cuda":
            try:
                cuda = cp.cuda.is_available()
            except Exception:
                cuda = False

            if not cuda:
                pytest.skip("Cuda is not available in the system")

        data = [1, 2, 3]
        tensor = Tensor(data, dtype=np.float32, device=device)
        assert_data(tensor, np.array(data, dtype=np.float32))
        assert tensor.device == device, (
            f"Tensor device {tensor.device} should be {device}"
        )

    def test_creation_with_grad(self):
        """Test tensor creation with gradient tracking enabled."""
        data = [1, 2, 3]
        tensor = Tensor(data, dtype=np.float32, requires_grad=True)
        assert_data(tensor, np.array(data, dtype=np.float32))
        assert tensor.grad is None, "Gradient should be None"
        assert tensor.requires_grad, "'requires_grad' should be True"


class TestTensorOperations:
    @pytest.mark.parametrize(
        "input_data, expected_output",
        [([1, -2, 3], [-1, 2, -3]), ([-1, 0, 2], [1, 0, -2])],
    )
    def test_unary_operations(self, input_data, expected_output):
        """Test unary operations like negation and absolute value."""
        tensor = Tensor(input_data, dtype=np.float32)

        neg_result = -tensor
        assert_data(neg_result, np.array(expected_output, dtype=np.float32))

        abs_result = abs(tensor)
        assert_data(abs_result, np.array(np.abs(input_data), dtype=np.float32))

    @pytest.mark.parametrize(
        "op_func, input1, input2, expected",
        [
            (
                lambda x, y: x + y,
                Tensor([1, 2, 3]),
                Tensor([4, 5, 6]),
                Tensor([5, 7, 9]),
            ),
            (lambda x, y: x + y, [1, 2, 3], Tensor([4, 5, 6]), Tensor([5, 7, 9])),
            (lambda x, y: x + y, Tensor([1, 2, 3]), [4, 5, 6], Tensor([5, 7, 9])),
            (
                lambda x, y: x - y,
                Tensor([4, 5, 6]),
                Tensor([1, 2, 3]),
                Tensor([3, 3, 3]),
            ),
            (lambda x, y: x - y, [4, 5, 6], Tensor([1, 2, 3]), Tensor([3, 3, 3])),
            (lambda x, y: x - y, Tensor([4, 5, 6]), [1, 2, 3], Tensor([3, 3, 3])),
            (
                lambda x, y: x * y,
                Tensor([1, 2, 3]),
                Tensor([4, 5, 6]),
                Tensor([4, 10, 18]),
            ),
            (lambda x, y: x * y, [1, 2, 3], Tensor([4, 5, 6]), Tensor([4, 10, 18])),
            (lambda x, y: x * y, Tensor([1, 2, 3]), [4, 5, 6], Tensor([4, 10, 18])),
            (
                lambda x, y: x / y,
                Tensor([4, 9, 16]),
                Tensor([2, 3, 4]),
                Tensor([2, 3, 4]),
            ),
            (lambda x, y: x / y, [4, 9, 16], Tensor([2, 3, 4]), Tensor([2, 3, 4])),
            (lambda x, y: x / y, Tensor([4, 9, 16]), [2, 3, 4], Tensor([2, 3, 4])),
        ],
    )
    def test_binary_operations(self, op_func, input1, input2, expected):
        """Comprehensive .benchmarks/test for tensor binary operations."""
        result = op_func(input1, input2)
        assert_data(result, expected)

    @pytest.mark.parametrize(
        "input_data, expected",
        [([1, 2, 3], np.exp([1, 2, 3])), ([0, -1, 2], np.exp([0, -1, 2]))],
    )
    def test_exponential_operations(self, input_data, expected):
        """Test exponential and logarithmic operations."""
        tensor = Tensor(input_data, dtype=np.float32)

        exp_result = op.exp(tensor)
        assert_data(exp_result, expected)

        log_result = op.log(Tensor([1, 2, 3], dtype=np.float32))
        assert_data(log_result, np.log([1, 2, 3]))

    @pytest.mark.parametrize(
        "input_data, expected",
        [
            ([0, 1, -1, 2, -2], np.tanh([0, 1, -1, 2, -2])),
            ([1000, -1000], np.tanh([1000, -1000])),
        ],
    )
    def test_activation_functions(self, input_data, expected):
        """Test activation functions like tanh."""
        tensor = Tensor(input_data, dtype=np.float32)
        result = op.tanh(tensor)
        assert_data(result, expected)


class TestTensorReduction:
    @pytest.mark.parametrize(
        "axis, keepdims, expected",
        [
            (None, False, [21]),
            (0, False, [5, 7, 9]),
            (1, False, [6, 15]),
            (None, True, [[21]]),
            (0, True, [[5, 7, 9]]),
            (1, True, [[6], [15]]),
        ],
    )
    def test_sum_variations(self, sample_2d_tensor, axis, keepdims, expected):
        """Comprehensive test for sum operation with various parameters."""
        result = sample_2d_tensor.sum(axis=axis, keepdims=keepdims)

        assert_data(result, np.array(expected, dtype=np.float32))

    @pytest.mark.parametrize(
        "axis, keepdims, expected",
        [
            (None, False, [6]),
            (0, False, [4, 5, 6]),
            (1, False, [3, 6]),
            (None, True, [[6]]),
            (0, True, [[4, 5, 6]]),
            (1, True, [[3], [6]]),
        ],
    )
    def test_max_variations(self, sample_2d_tensor, axis, keepdims, expected):
        """Comprehensive test for max operation with various parameters."""
        result = sample_2d_tensor.max(axis=axis, keepdims=keepdims)

        assert_data(result, np.array(expected, dtype=np.float32))


class TestTensorShapeOperations:
    @pytest.mark.parametrize(
        "target_shape, expected_data",
        [
            ((3, 2), [1, 2, 3, 4, 5, 6]),
            ((6,), [1, 2, 3, 4, 5, 6]),
            ((1, 2, 3), [1, 2, 3, 4, 5, 6]),
        ],
    )
    def test_reshape(self, sample_2d_tensor, target_shape, expected_data):
        """Test reshape with various target shapes."""
        result = sample_2d_tensor.reshape(target_shape)
        assert_data(
            result, np.array(expected_data, dtype=np.float32).reshape(target_shape)
        )
        assert result.shape == target_shape, (
            f"Result shape {result.shape} should be {target_shape}, Error in reshape"
        )

    @pytest.mark.parametrize(
        "invalid_shape",
        [
            (4, 2),
            (2,),
            (3,),
        ],
    )
    def test_reshape_error_cases(
        self, sample_2d_tensor, invalid_shape: tuple[int, ...]
    ):
        """Test reshape with invalid shapes."""
        with pytest.raises(ValueError):
            sample_2d_tensor.reshape(invalid_shape)

    def test_transpose(self, sample_2d_tensor):
        """Test tensor transpose operation."""
        result = op.transpose(sample_2d_tensor)
        assert_data(result, np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32))

    def test_matmul(self):
        """Test matrix multiplication with various input shapes."""
        tensor1 = Tensor([[1, 2], [3, 4]], dtype=np.float32)
        tensor2 = Tensor([[5, 6], [7, 8]], dtype=np.float32)

        result = tensor1 @ tensor2
        assert_data(result, np.array([[19, 22], [43, 50]], dtype=np.float32))

        vector = Tensor([5, 6], dtype=np.float32)
        vector_result = tensor1 @ vector
        assert_data(vector_result, np.array([17, 39], dtype=np.float32))

    def test_flatten(self):
        """Test tensor flattening."""
        tensor = Tensor([[1, 2], [3, 4]], dtype=np.float32)

        t1 = tensor.flatten()

        assert_data(t1, tensor.data.flatten())


class TestTensorProperties:
    def test_tensor_properties(self, sample_2d_tensor):
        """Test basic tensor properties."""
        assert sample_2d_tensor.shape == (2, 3), "Shape should be (2, 3)"
        assert sample_2d_tensor.size == 6, "Size should be 6"
        assert sample_2d_tensor.dtype == np.float32, "Data type should be np.float32"

    def test_grad_management(self):
        """Test gradient management for tensors."""
        tensor = Tensor([1, 2, 3], dtype=np.float32, requires_grad=True)
        tensor.clear_grad()
        assert tensor.grad is None

        tensor.grad = np.zeros_like(tensor.data)
        tensor.clear_grad()
        assert_grad(tensor, np.zeros_like(tensor.data))


class TestTensorMiscOperations:
    def test_rounding(self):
        """Test tensor rounding operation."""
        tensor = Tensor([1.1, 2.5, 3.7], dtype=np.float32)
        result = op.round(tensor)
        assert_data(result, np.round(np.array([1.1, 2.5, 3.7], dtype=np.float32)))
