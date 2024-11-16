import numpy as np

from tensor import Tensor
from tests.utils import assert_data, assert_grad


def test_tensor_creation():
    data = [1, 2, 3]
    tensor = Tensor(data, dtype=np.float32)
    assert_data(tensor, np.array(data, dtype=np.float32))
    assert not tensor.requires_grad


def test_tensor_creation_with_grad():
    data = [1, 2, 3]
    tensor = Tensor(data, dtype=np.float32, requires_grad=True)
    assert_data(tensor, np.array(data, dtype=np.float32))
    assert_grad(tensor, np.zeros_like(tensor.data))
    assert tensor.requires_grad


def test_tensor_negation():
    tensor = Tensor([1, -2, 3], dtype=np.float32)
    result = -tensor
    assert_data(result, np.array([-1, 2, -3], dtype=np.float32))


def test_tensor_absolute_value():
    tensor = Tensor([-1, -2, 3], dtype=np.float32)
    result = abs(tensor)
    assert_data(result, np.array([1, 2, 3], dtype=np.float32))


def test_tensor_addition():
    tensor1 = Tensor([1, 2, 3], dtype=np.float32)
    tensor2 = Tensor([4, 5, 6], dtype=np.float32)
    result = tensor1 + tensor2
    assert_data(result, np.array([5, 7, 9], dtype=np.float32))


def test_tensor_subtraction():
    tensor1 = Tensor([4, 5, 6], dtype=np.float32)
    tensor2 = Tensor([1, 2, 3], dtype=np.float32)
    result = tensor1 - tensor2
    assert_data(result, np.array([3, 3, 3], dtype=np.float32))


def test_tensor_multiplication():
    tensor1 = Tensor([1, 2, 3], dtype=np.float32)
    tensor2 = Tensor([4, 5, 6], dtype=np.float32)
    result = tensor1 * tensor2
    assert_data(result, np.array([4, 10, 18], dtype=np.float32))


def test_tensor_division():
    tensor1 = Tensor([4, 9, 16], dtype=np.float32)
    tensor2 = Tensor([2, 3, 4], dtype=np.float32)
    result = tensor1 / tensor2
    assert_data(result, np.array([2, 3, 4], dtype=np.float32))


def test_tensor_power():
    tensor1 = Tensor([2, 3, 4], dtype=np.float32)
    tensor2 = Tensor([3, 2, 1], dtype=np.float32)
    result = tensor1**tensor2
    assert_data(result, np.array([8, 9, 4], dtype=np.float32))


def test_value_addition_to_tensor():
    tensor = Tensor([1, 2, 3], dtype=np.float32)
    value = 5
    result = value + tensor
    assert_data(result, np.array([6, 7, 8], dtype=np.float32))


def test_value_subtraction_from_tensor():
    tensor = Tensor([4, 5, 6], dtype=np.float32)
    value = 2
    result = value - tensor
    assert_data(result, np.array([-2, -3, -4], dtype=np.float32))


def test_value_multiplication_with_tensor():
    tensor = Tensor([1, 2, 3], dtype=np.float32)
    value = 3
    result = value * tensor
    assert_data(result, np.array([3, 6, 9], dtype=np.float32))


def test_value_division_by_tensor():
    tensor = Tensor([2, 4, 8], dtype=np.float32)
    value = 16
    result = value / tensor
    assert_data(result, np.array([8, 4, 2], dtype=np.float32))


def test_value_power_of_tensor():
    tensor = Tensor([2, 3, 4], dtype=np.float32)
    value = 2
    result = value**tensor
    assert_data(result, np.array([4, 8, 16], dtype=np.float32))


def test_tensor_inplace_addition():
    tensor1 = Tensor([1, 2, 3], dtype=np.float32)
    tensor2 = Tensor([4, 5, 6], dtype=np.float32)
    tensor1 += tensor2
    assert_data(tensor1, np.array([5, 7, 9], dtype=np.float32))


def test_tensor_inplace_subtraction():
    tensor1 = Tensor([4, 5, 6], dtype=np.float32)
    tensor2 = Tensor([1, 2, 3], dtype=np.float32)
    tensor1 -= tensor2
    assert_data(tensor1, np.array([3, 3, 3], dtype=np.float32))


def test_tensor_inplace_multiplication():
    tensor1 = Tensor([1, 2, 3], dtype=np.float32)
    tensor2 = Tensor([4, 5, 6], dtype=np.float32)
    tensor1 *= tensor2
    assert_data(tensor1, np.array([4, 10, 18], dtype=np.float32))


def test_tensor_inplace_division():
    tensor1 = Tensor([4, 9, 16], dtype=np.float32)
    tensor2 = Tensor([2, 3, 4], dtype=np.float32)
    tensor1 /= tensor2
    assert_data(tensor1, np.array([2, 3, 4], dtype=np.float32))


def test_tensor_inplace_power():
    tensor1 = Tensor([2, 3, 4], dtype=np.float32)
    tensor2 = Tensor([3, 2, 1], dtype=np.float32)
    tensor1 **= tensor2
    assert_data(tensor1, np.array([8, 9, 4], dtype=np.float32))


def test_tensor_shape():
    tensor = Tensor([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    assert tensor.shape == (2, 3)


def test_tensor_size():
    tensor = Tensor([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    assert tensor.size == 6


def test_tensor_dtype():
    tensor = Tensor([1, 2, 3], dtype=np.float32)
    assert tensor.dtype == np.float32


def test_tensor_clear_grad():
    tensor = Tensor([1, 2, 3], dtype=np.float32, requires_grad=True)
    tensor.clear_grad()
    assert_grad(tensor, np.zeros_like(tensor.data))
