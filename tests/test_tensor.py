from typing import Callable

import cupy as cp
import numpy as np
from numpy.typing import NDArray, DTypeLike
import pytest

from src.tensor import Tensor, op
from .utils import assert_data, assert_grad


@pytest.fixture
def sample_2d_tensor() -> Tensor[np.float32]:
    """Fixture for a standard 2D tensor used across multiple tests."""
    return Tensor([[1, 2, 3], [4, 5, 6]], dtype=np.float32)


class TestTensorCreation:
    def test_basic_creation(self) -> None:
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
    def test_cpu_creation(self, device: str) -> None:
        """Test tensor creation on devices."""
        if device == "cuda":
            try:
                cuda = cp.cuda.is_available()
            except Exception:
                cuda = False

            if not cuda:
                pytest.skip(
                    "Cuda is not available in the system. Make sure to run the test on a system with CUDA support."
                )

        data = [1, 2, 3]
        tensor = Tensor(data, dtype=np.float32, device=device)
        assert_data(tensor, np.array(data, dtype=np.float32))
        assert tensor.device == device, f"Tensor device {tensor.device} should be {device}"

    def test_creation_with_grad(self) -> None:
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
    def test_unary_operations(self, input_data: list[int], expected_output: list[int]) -> None:
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
    def test_binary_operations(
        self,
        op_func: Callable,
        input1: list[int] | Tensor[np.integer],
        input2: list[int] | Tensor[np.integer],
        expected: Tensor,
    ) -> None:
        """Tests for tensor binary operations."""
        result = op_func(input1, input2)
        assert_data(result, expected)

    @pytest.mark.parametrize(
        "input_data, expected",
        [([1, 2, 3], np.exp([1, 2, 3])), ([0, -1, 2], np.exp([0, -1, 2]))],
    )
    def test_exponential_operations(self, input_data: list[int], expected: NDArray) -> None:
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
    def test_activation_functions(self, input_data: list[int], expected: NDArray):
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
    def test_sum_variations(self, sample_2d_tensor: Tensor, axis: int | None, keepdims: bool, expected: list) -> None:
        """Test for sum operation with various parameters."""
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
    def test_max_variations(self, sample_2d_tensor: Tensor, axis: int | None, keepdims: bool, expected: list) -> None:
        """Test for max operation with various parameters."""
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
    def test_reshape(self, sample_2d_tensor: Tensor, target_shape: tuple[int, ...], expected_data: list) -> None:
        """Test reshape with various target shapes."""

        result = op.reshape(sample_2d_tensor, target_shape)

        assert result is not sample_2d_tensor, "Result tensor should be different from input tensor"

        assert_data(result, np.array(expected_data, dtype=np.float32).reshape(target_shape))

        assert result.shape == target_shape, f"Result shape {result.shape} should be {target_shape}, Error in reshape"

    def test_reshape_inplace(self, sample_2d_tensor: Tensor) -> None:
        """Test reshape operation inplace."""
        tensor = sample_2d_tensor.copy()

        result = tensor.reshape((3, 2), inplace=True)

        assert tensor is result, "Result tensor should be the same as input tensor"
        assert result.shape == (3, 2), "Shape should be (3, 2)"

    @pytest.mark.parametrize(
        "invalid_shape",
        [
            (4, 2),
            (2,),
            (3,),
        ],
    )
    def test_reshape_error_cases(self, sample_2d_tensor: Tensor, invalid_shape: tuple[int, ...]) -> None:
        """Test reshape with invalid shapes."""
        with pytest.raises(ValueError):
            sample_2d_tensor.reshape(invalid_shape)

    def test_transpose(self, sample_2d_tensor: Tensor):
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

    @pytest.mark.parametrize(
        "data, shape, strides, operation, grad",
        [
            ([1, 2, 3], (3, 1), (4, 4), lambda x: x[1] * 2, [0, 2, 0]),
            ([1, 2, 3, 4, 5], (2, 2), (4, 4), lambda x: x.sum(), [1, 2, 1, 0, 0]),
            ([1, 2, 3, 4], (2, 2), (8, 4), lambda x: (x * x).sum(), [2, 4, 6, 8]),
            ([7], (3, 1), (0, 0), lambda x: x.sum(), [3]),
            ([[1, 2], [3, 4]], (2, 2, 1), (8, 4, 4), lambda x: x[0, 1, 0] * 2, [[0, 2], [0, 0]]),
            ([1, 2, 3, 4, 5, 6], (3,), (8,), lambda x: x[0] + x[1] + x[2], [1, 0, 1, 0, 1, 0]),
            ([1, 2, 3, 4, 5, 6, 7, 8, 9], (3, 3), (12, 4), lambda x: x[1, 1] * 3, [0, 0, 0, 0, 3, 0, 0, 0, 0]),
            ([1, 10, 2, 20, 3, 30], (3, 2), (8, 4), lambda x: x[:, 0].sum(), [1, 0, 1, 0, 1, 0]),
        ],
        ids=(
            "simple_1d_to_2d",
            "overlapping_elements",
            "matrix_like_strides",
            "broadcasting",
            "2d_to_3d",
            "skip_elements",
            "complex_view",
            "interleaved_data",
        ),
    )
    def test_as_strides(self, data, shape, strides, operation, grad):
        """Test tensor as_strides operation."""
        tensor = Tensor(data, requires_grad=True, dtype=np.float32, device="cpu")

        result = op.as_strided(tensor, shape=shape, strides=strides)

        assert_data(
            result, np.lib.stride_tricks.as_strided(np.array(data, dtype=np.float32), shape=shape, strides=strides)
        )
        assert result.shape == shape, f"Shape should be {shape}"
        assert result.strides == strides, f"Strides should be {strides}"

        operation(result).backward()

        assert_grad(tensor, np.array(grad))

    def test_windows_as_strides(self):
        """Test tensor windows_as_strides operation."""
        shape = (1, 2, 30, 30)
        num_items = np.prod(shape)
        data = Tensor(np.arange(num_items).reshape(shape), dtype=np.float32, requires_grad=True, device="cpu")
        window_shape = (3, 3)

        shape = (1, 1, 28, 28, 2, *window_shape)
        strides = (
            data.strides[0],
            0,
            data.strides[2],
            data.strides[3],
            data.strides[1],
            data.strides[2],
            data.strides[3],
        )
        windows = op.as_strided(data, shape=shape, strides=strides)

        expected_windows = np.lib.stride_tricks.sliding_window_view(data, (1, 2, *window_shape)).reshape(shape)
        assert_data(windows, expected_windows)

        windows.backward()

        assert data.grad is not None


class TestTensorProperties:
    def test_tensor_properties(self, sample_2d_tensor: Tensor) -> None:
        """Test basic tensor properties."""
        assert sample_2d_tensor.shape == (2, 3), "Shape should be (2, 3)"
        assert sample_2d_tensor.size == 6, "Size should be 6"
        assert sample_2d_tensor.dtype == np.float32, "Data type should be np.float32"

    def test_grad_management(self) -> None:
        """Test gradient management for tensors."""
        tensor = Tensor([1, 2, 3], dtype=np.float32, requires_grad=True)
        assert tensor.grad is None

        tensor.grad = np.zeros_like(tensor.data)
        tensor.zero_grad()
        assert tensor.grad is None


class TestTensorMiscOperations:
    def test_rounding(self) -> None:
        """Test tensor rounding operation."""
        data = np.array([1.1, 2.5, 3.7], dtype=np.float32)
        tensor = Tensor(data)
        result = op.round(tensor)

        assert result is not tensor, "Result tensor should be different from input tensor"

        assert_data(tensor, data, "Input tensor should not be modified")
        assert_data(result, np.round(np.array([1.1, 2.5, 3.7], dtype=np.float32)))

    def test_round_inplace(self) -> None:
        """Test tensor rounding operation inplace."""
        tensor = Tensor([1.1, 2.5, 3.7], dtype=np.float32)
        result = op.round(tensor, inplace=True)

        assert result is tensor, "Result tensor should be the same as input tensor"

        assert_data(tensor, np.round(np.array([1.1, 2.5, 3.7], dtype=np.float32)))


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        np.int32,
        np.int64,
        cp.float32,
        cp.float64,
        cp.int32,
        cp.int64,
        float,
        int,
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_config(dtype: np.dtype | cp.dtype | DTypeLike, device: str) -> None:
    """Test tensor creation with custom configuration."""
    Tensor.set_default_dtype(dtype)
    Tensor.set_default_device(device)
    data = [1, 2, 3]
    tensor = Tensor(data)
    assert_data(tensor, np.array(data, dtype=Tensor.default_dtype))
    assert tensor.device == Tensor.default_device, f"Tensor device {tensor.device} should be {Tensor.default_device}"
    assert tensor.dtype == Tensor.default_dtype, f"Data type should be {Tensor.default_dtype}. Got {tensor.dtype}"

    Tensor.set_default_device("auto")
