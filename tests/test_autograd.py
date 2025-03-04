import numpy as np

from src.core import Tensor
from src.core.tensor import op
from src.constants import EPSILON

from .utils import assert_grad


class TestUnaryOperations:
    def test_neg_backward(self):
        """Test the backward computation for the negation operation."""

        a = Tensor([1, -2, 3], dtype=np.float32, requires_grad=True)
        b = -a

        b.backward()

        assert_grad(a, -np.ones_like(a))

    def test_abs_backward(self):
        """Test the backward computation for the absolute value operation."""

        a = Tensor([-1, 0, 3], dtype=np.float32, requires_grad=True)
        b = abs(a)

        b.backward()

        # -1 for negative, 1 for positive, 0 for 0
        expected_grad_a = ((a.data > 0) - 1 * (a.data < 0)) * (a.data != 0)

        assert_grad(a, expected_grad_a)


class TestBinaryOperations:
    def test_add_backward(self):
        """Test the backward computation for the addition operation."""

        a = Tensor([1, 2, 3], dtype=np.float32, requires_grad=True)
        b = Tensor([4, 5, 6], dtype=np.float32, requires_grad=True)
        c = a + b

        c.backward()

        assert_grad(a, np.ones_like(a))
        assert_grad(b, np.ones_like(b))

    def test_add_backward_one_element_tensor(self):
        """Test the backward computation for the addition operation with one element tensor."""

        a = Tensor([1, 2, 3], dtype=np.float32, requires_grad=True)
        b = Tensor([2], dtype=np.float32, requires_grad=True)
        c = a + b

        c.backward()

        assert_grad(a, np.ones_like(a))
        assert_grad(b, np.sum(np.ones_like(a)))

        a.clear_grad()
        b.clear_grad()
        c.clear_grad()

        c = b + a

        c.backward()

        assert_grad(a, np.ones_like(a))
        assert_grad(b, np.sum(np.ones_like(a)))

    def test_mul_backward(self):
        """Test the backward computation for the multiplication operation."""

        a = Tensor([1, 2, 3], dtype=np.float32, requires_grad=True)
        b = Tensor([4, 5, 6], dtype=np.float32, requires_grad=True)
        c = a * b

        c.backward()

        assert_grad(a, b.data)
        assert_grad(b, a.data)

    def test_mul_backward_one_element_tensor(self):
        """Test the backward computation for the multiplication operation with one element tensor."""

        a = Tensor([1, 2, 3], dtype=np.float32, requires_grad=True)
        b = Tensor([2], dtype=np.float32, requires_grad=True)
        c = a * b

        c.backward()

        assert_grad(a, b.data)
        assert_grad(b, np.sum(a.data))

        a.clear_grad()
        b.clear_grad()
        c.clear_grad()

        c = b * a

        c.backward()

        assert_grad(a, b.data)
        assert_grad(b, np.sum(a.data))

    def test_sub_backward(self):
        """Test the backward computation for the subtract operation."""

        a = Tensor([4, 5, 6], dtype=np.float32, requires_grad=True)
        b = Tensor([1, 2, 3], dtype=np.float32, requires_grad=True)
        c = a - b

        c.backward()

        assert_grad(a, np.ones_like(a))
        assert_grad(b, np.full_like(b, -1))

    def test_sub_backward_one_element_tensor(self):
        """Test the backward computation for the subtract operation with one element tensor."""

        a = Tensor([4, 5, 6], dtype=np.float32, requires_grad=True)
        b = Tensor([2], dtype=np.float32, requires_grad=True)
        c = a - b

        c.backward()

        assert_grad(a, np.ones_like(a))
        assert_grad(b, -np.sum(np.ones_like(a)))

        a.clear_grad()
        b.clear_grad()
        c.clear_grad()

        c = b - a

        c.backward()

        assert_grad(a, np.full_like(a, -1))
        assert_grad(b, np.sum(np.ones_like(a)))

    def test_div_backward(self):
        """Test the backward computation for the division operation."""

        a = Tensor([4, 6, 8], dtype=np.float32, requires_grad=True)
        b = Tensor([2, 2, 2], dtype=np.float32, requires_grad=True)
        c = a / b

        c.backward()

        assert_grad(a, 1 / b.data)
        assert_grad(b, -a.data / (b.data**2))

    def test_div_backward_one_element_tensor(self):
        """Test the backward computation for the division operation with one element tensor."""

        a = Tensor([2, 4, 6, 9], dtype=np.float32, requires_grad=True)
        b = Tensor([2], dtype=np.float32, requires_grad=True)
        c = a / b

        c.backward()

        assert_grad(a, 1 / b.data)
        assert_grad(b, np.sum(-a.data / (b.data**2)))

        a.clear_grad()
        b.clear_grad()
        c.clear_grad()

        c = b / a

        c.backward()

        assert_grad(a, -b.data / (a.data**2))
        assert_grad(b, np.sum(1 / a.data))

    def test_matmul_backward(self):
        """Test the backward computation for the matrix multiplication operation."""

        a = Tensor([[1, 2], [3, 4]], dtype=np.float32, requires_grad=True)
        b = Tensor([[5, 6], [7, 8]], dtype=np.float32, requires_grad=True)
        c = a @ b

        c.backward()

        expected_grad_a = np.dot(np.ones_like(c.data), b.data.T)
        expected_grad_b = np.dot(a.data.T, np.ones_like(c.data))

        assert_grad(a, expected_grad_a)
        assert_grad(b, expected_grad_b)

    def test_pow_backward(self):
        """Test the backward computation for the power operation."""

        a = Tensor([2, 3, 4], dtype=np.float32, requires_grad=True)
        b = Tensor([2, 2, 2], dtype=np.float32, requires_grad=True)
        c = a**b

        c.backward()

        expected_grad_a = b.data * (a.data ** (b.data - 1))
        expected_grad_b = (a.data**b.data) * np.log(a.data)

        assert_grad(a, expected_grad_a)
        assert_grad(b, expected_grad_b)

    def test_pow_backward_one_element_tensor(self):
        """Test the backward computation for the power operation with one element tensor."""

        a = Tensor([2, 3, 4], dtype=np.float32, requires_grad=True)
        b = Tensor([2], dtype=np.float32, requires_grad=True)
        c = a**b

        c.backward()

        expected_grad_a = b.data * (a.data ** (b.data - 1))
        expected_grad_b = np.sum((a.data**b.data) * np.log(a.data))

        assert_grad(a, expected_grad_a)
        assert_grad(b, expected_grad_b)

        a.clear_grad()
        b.clear_grad()
        c.clear_grad()

        c = b**a

        c.backward()

        expected_grad_a = (b.data**a.data) * np.log(b.data)
        expected_grad_b = np.sum(a.data * (b.data ** (a.data - 1)))

        assert_grad(a, expected_grad_a)
        assert_grad(b, expected_grad_b)


class TestFunctionalOperations:
    def test_exp_backward(self):
        """Test the backward computation for the exponential operation."""

        a = Tensor([1, 2, 3], dtype=np.float32, requires_grad=True)
        b = op.exp(a)

        b.backward()

        expected_grad_a = np.exp(a.data)

        assert_grad(a, expected_grad_a)

    def test_sum_backward(self):
        """Test the backward computation for the sum operation."""

        a = Tensor([[1, 2, 3], [4, 5, 6]], dtype=np.float32, requires_grad=True)
        b = op.sum(a)

        b.backward()

        # For sum, backward should be 1 for all elements
        expected_grad_a = np.ones_like(a.data)

        assert_grad(a, expected_grad_a)

    def test_sum_backward_axis_0(self):
        """Test the backward computation for the sum operation along axis 0."""

        a = Tensor([[1, 2, 3], [4, 5, 6]], dtype=np.float32, requires_grad=True)
        b = op.sum(a, axis=0)

        b.backward()

        # For sum along axis 0, backward should be 1 for all elements in the summed axis
        expected_grad_a = np.ones_like(a.data)

        assert_grad(a, expected_grad_a)

    def test_sum_backward_axis_1(self):
        """Test the backward computation for the sum operation along axis 1."""

        a = Tensor([[1, 2, 3], [4, 5, 6]], dtype=np.float32, requires_grad=True)
        b = op.sum(a, axis=1)

        b.backward()

        # For sum along axis 1, backward should be 1 for all elements in the summed axis
        expected_grad_a = np.ones_like(a.data)

        assert_grad(a, expected_grad_a)

    def test_sum_backward_keepdims(self):
        """Test the backward computation for the sum operation with keepdims=True."""

        a = Tensor([[1, 2, 3], [4, 5, 6]], dtype=np.float32, requires_grad=True)
        b = a.sum(keepdims=True)

        b.backward()

        # For sum with keepdims=True, backward should be 1 for all elements
        expected_grad_a = np.ones_like(a.data)

        assert_grad(a, expected_grad_a)

    def test_sum_backward_axis_0_keepdims(self):
        """Test the backward computation for the sum operation along axis 0 with keepdims=True."""

        a = Tensor([[1, 2, 3], [4, 5, 6]], dtype=np.float32, requires_grad=True)
        b = op.sum(a, axis=0, keepdims=True)

        b.backward()

        # For sum along axis 0 with keepdims=True, backward should be 1 for all elements in the summed axis
        expected_grad_a = np.ones_like(a.data)

        assert_grad(a, expected_grad_a)

    def test_sum_backward_axis_1_keepdims(self):
        """Test the backward computation for the sum operation along axis 1 with keepdims=True."""

        a = Tensor([[1, 2, 3], [4, 5, 6]], dtype=np.float32, requires_grad=True)
        b = op.sum(a, axis=1, keepdims=True)

        b.backward()

        # For sum along axis 1 with keepdims=True, backward should be 1 for all elements in the summed axis
        expected_grad_a = np.ones_like(a.data)

        assert_grad(a, expected_grad_a)

    def test_max_backward(self):
        """Test the backward computation for the max operation."""

        a = Tensor([1, 3, 2], dtype=np.float32, requires_grad=True)
        b = a.max()

        b.backward()

        # For max, backward is 1 for the max element and 0 for others
        expected_grad_a = np.zeros_like(a.data)
        expected_grad_a[np.argmax(a.data)] = 1

        assert_grad(a, expected_grad_a)

    def test_max_backward_axis_0(self):
        """Test the backward computation for the max operation along axis 0."""

        a = Tensor([[1, 4, 3], [2, 3, 5]], dtype=np.float32, requires_grad=True)
        b = a.max(axis=0)

        b.backward()

        # For max along axis 0, backward is 1 for the max elements in the axis and 0 for others
        expected_grad_a = np.zeros_like(a.data)
        expected_grad_a[np.argmax(a.data, axis=0), np.arange(a.shape[1])] = 1

        assert_grad(a, expected_grad_a)

    def test_max_backward_axis_1(self):
        """Test the backward computation for the max operation along axis 1."""

        a = Tensor([[1, 4, 3], [2, 3, 5]], dtype=np.float32, requires_grad=True)
        b = a.max(axis=1)

        b.backward()

        # For max along axis 1, backward is 1 for the max elements in the axis and 0 for others
        expected_grad_a = np.zeros_like(a.data)
        expected_grad_a[np.arange(a.shape[0]), np.argmax(a.data, axis=1)] = 1

        assert_grad(a, expected_grad_a)

    def test_max_backward_keepdims(self):
        """Test the backward computation for the max operation with keepdims=True."""

        a = Tensor([[1, 4, 3], [2, 3, 5]], dtype=np.float32, requires_grad=True)
        b = a.max(keepdims=True)

        b.backward()

        # For max with keepdims=True, backward is 1 for the max elements and 0 for others
        expected_grad_a = np.zeros_like(a.data)
        expected_grad_a[np.unravel_index(np.argmax(a.data), a.shape)] = 1

        assert_grad(a, expected_grad_a)

    def test_max_backward_axis_0_keepdims(self):
        """Test the backward computation for the max operation along axis 0 with keepdims=True."""

        a = Tensor([[1, 4, 3], [2, 3, 5]], dtype=np.float32, requires_grad=True)
        b = a.max(axis=0, keepdims=True)

        b.backward()

        # For max along axis 0 with keepdims=True, backward is 1 for the max elements in the axis and 0 for others
        expected_grad_a = np.zeros_like(a.data)
        expected_grad_a[np.argmax(a.data, axis=0), np.arange(a.shape[1])] = 1

        assert_grad(a, expected_grad_a)

    def test_max_backward_axis_1_keepdims(self):
        """Test the backward computation for the max operation along axis 1 with keepdims=True."""

        a = Tensor([[1, 4, 3], [2, 3, 5]], dtype=np.float32, requires_grad=True)
        b = a.max(axis=1, keepdims=True)

        b.backward()

        # For max along axis 1 with keepdims=True, backward is 1 for the max elements in the axis and 0 for others
        expected_grad_a = np.zeros_like(a.data)
        expected_grad_a[np.arange(a.shape[0]), np.argmax(a.data, axis=1)] = 1

        assert_grad(a, expected_grad_a)

    def test_mean_backwards(self):
        """Test the backward computation for the mean operation."""
        a = Tensor([1, 3, 2], dtype=np.float32, requires_grad=True)
        b = a.mean()

        b.backward()

        # For mean, backward distributes gradient evenly (1/3 for each element)
        expected_grad_a = np.ones_like(a.data) / len(a.data)

        assert_grad(a, expected_grad_a)

    def test_mean_backward_axis_0(self):
        """Test the backward computation for the mean operation along axis 0."""
        a = Tensor([[1, 4, 3], [2, 3, 5]], dtype=np.float32, requires_grad=True)
        b = a.mean(axis=0)

        b.backward()

        expected_grad_a = 1 / a.data.shape[0]

        assert_grad(a, expected_grad_a)

    def test_mean_backward_axis_0_1(self):
        """Test the backward computation for the mean operation along axis 1.2."""
        a = Tensor(
            [[1, 4, 3], [2, 3, 5], [6, 7, 8]], dtype=np.float32, requires_grad=True
        )
        b = a.mean(axis=(0, 1))

        b.backward()

        expected_grad_a = np.ones_like(a.data) / (a.data.shape[0] * a.data.shape[1])

        assert_grad(a, expected_grad_a)

    def test_tanh_backward(self):
        """Test the backward computation for the tanh operation."""

        a = Tensor([0.5, -1.0, 2.0], dtype=np.float32, requires_grad=True)
        b = op.tanh(a)

        b.backward()

        # For tanh, backward is 1 - tanh^2(a)
        expected_grad_a = 1 - np.tanh(a.data) ** 2

        assert_grad(a, expected_grad_a)

    def test_log_backward(self):
        """Test the backward computation for the log operation."""

        a = Tensor([1, 2, 3], dtype=np.float32, requires_grad=True)
        b = op.log(a)

        b.backward()

        # For log, backward is 1/a
        expected_grad_a = 1 / a.data

        assert_grad(a, expected_grad_a)

    def test_reshape_backward(self):
        """Test the backward computation for the reshape operation."""

        a = Tensor([[1, 2, 3], [4, 5, 6]], dtype=np.float32, requires_grad=True)
        b = a.reshape((3, 2), inplace=False)

        b.backward()

        # For reshape, backward is the same as the original tensor
        assert_grad(a, np.ones_like(a))

    def test_flatten_backward(self):
        """Test the backward computation for the flatten operation."""
        a = Tensor([[1, 2, 3], [4, 5, 6]], dtype=np.float32, requires_grad=True)
        b = a.flatten()

        b.backward()

        assert_grad(a, np.ones_like(a))

    def test_expand_dims_backward(self):
        """Test the backward computation for the expand dimensions operation."""
        a = Tensor([[1, 2, 3], [4, 5, 6]], dtype=np.float32, requires_grad=True)
        b = op.expand_dims(a, 0)

        b.backward()

        assert_grad(a, np.ones_like(a))

    def test_round_backward(self):
        """Test the backward computation for the round operation."""

        a = Tensor([1.2, 2.5, 3.7], dtype=np.float32, requires_grad=True)
        b = op.round(a)

        b.backward()

        expected_grad_a = np.ones_like(a.data)

        assert_grad(a, expected_grad_a)

    def test_exponential_operations(self):
        """Test the backward computation for the exponential operation"""
        a = Tensor([1, 2, 3], dtype=np.float32, requires_grad=True)
        b = Tensor([2, 3, 4], dtype=np.float32, requires_grad=True)
        c = a**b
        d = op.exp(c)

        d.backward()

        grad_c = d.data
        grad_b = grad_c * (a.data**b.data) * np.log(a.data + EPSILON)
        grad_a = grad_c * b.data * (a.data ** (b.data - 1))

        assert_grad(c, grad_c)
        assert_grad(b, grad_b)
        assert_grad(a, grad_a)


class TestComplexChainOperations:
    def test_sum_and_max_chain(self):
        """Test the backward computation for a chain of sum and max operations."""

        a = Tensor([[1, 2, 3], [4, 5, 6]], dtype=np.float32, requires_grad=True)
        b = op.sum(a, axis=1)  # [6, 15]
        c = b.max()  # 15

        c.backward()

        expected_grad_b = np.zeros_like(b.data)
        expected_grad_b[np.argmax(b.data)] = 1

        expected_grad_a = np.zeros_like(a.data)
        expected_grad_a[np.argmax(b.data), :] = 1

        assert_grad(b, expected_grad_b)
        assert_grad(a, expected_grad_a)

    def test_complex_sum_and_max_chain(self):
        """Test the backward computation for a complex chain of sum and max operations."""

        a = Tensor([[1, 2, 3], [4, 5, 6]], dtype=np.float32, requires_grad=True)
        b = op.sum(a, axis=0)
        c = b.max()
        d = c + op.sum(a)
        e = d * op.max(a)

        e.backward()

        expected_grad_b = np.zeros_like(b.data)
        expected_grad_b[np.argmax(b.data)] = op.max(a).data

        expected_grad_a = np.zeros_like(a.data)
        expected_grad_a[:, np.argmax(b.data)] = op.max(a).data

        expected_grad_a += np.ones_like(a.data) * 6

        expected_grad_a[np.unravel_index(np.argmax(a.data), a.shape)] += 30

        assert_grad(b, expected_grad_b)
        assert_grad(a, expected_grad_a)
