import numpy as np

from tensor import Tensor
import tensor.op as op

from tests.utils import assert_grad


def test_add_backward():
    """Test the backward computation for the addition operation."""

    a = Tensor([1, 2, 3], dtype=np.float32, requires_grad=True)
    b = Tensor([4, 5, 6], dtype=np.float32, requires_grad=True)
    c = a + b

    c.gradient()

    # For addition, gradient should be 1 for both inputs
    assert_grad(a, np.ones_like(a))
    assert_grad(b, np.ones_like(b))


def test_mul_backward():
    """Test the backward computation for the multiplication operation."""

    a = Tensor([1, 2, 3], dtype=np.float32, requires_grad=True)
    b = Tensor([4, 5, 6], dtype=np.float32, requires_grad=True)
    c = a * b

    c.gradient()

    # For multiplication, gradient of each input is the other input
    assert_grad(a, b.data)
    assert_grad(b, a.data)


def test_sub_backward():
    """Test the backward computation for the subtract operation."""

    a = Tensor([4, 5, 6], dtype=np.float32, requires_grad=True)
    b = Tensor([1, 2, 3], dtype=np.float32, requires_grad=True)
    c = a - b

    c.gradient()

    # For subtraction, gradient is 1 for first input and -1 for second input
    assert_grad(a, np.ones_like(a))
    assert_grad(b, np.full_like(b, -1))


def test_div_backward():
    """Test the backward computation for the division operation."""

    a = Tensor([4, 6, 8], dtype=np.float32, requires_grad=True)
    b = Tensor([2, 2, 2], dtype=np.float32, requires_grad=True)
    c = a / b

    c.gradient()

    # For division, gradient of first input is 1/b, gradient of second input is -a/(b^2)
    assert_grad(a, 1 / b.data)
    assert_grad(b, -a.data / (b.data**2))


def test_pow_backward():
    """Test the backward computation for the power operation."""

    a = Tensor([2, 3, 4], dtype=np.float32, requires_grad=True)
    b = Tensor([2, 2, 2], dtype=np.float32, requires_grad=True)
    c = a**b

    c.gradient()

    # For power, gradient of base is exponent * base^(exponent-1)
    # Gradient of exponent is base^exponent * ln(base)
    expected_grad_a = b.data * (a.data ** (b.data - 1))
    expected_grad_b = (a.data**b.data) * np.log(a.data)

    assert_grad(a, expected_grad_a)
    assert_grad(b, expected_grad_b)


def test_exp_backward():
    """Test the backward computation for the exponential operation."""

    a = Tensor([1, 2, 3], dtype=np.float32, requires_grad=True)
    b = op.exp(a)

    b.gradient()

    expected_grad_a = np.exp(a.data)

    assert_grad(a, expected_grad_a)


def test_requires_grad():
    """Test that the requires_grad attribute is set correctly for operations."""

    a = Tensor([1, 2, 3], dtype=np.float32, requires_grad=False)
    b = Tensor([4, 5, 6], dtype=np.float32, requires_grad=True)

    c = a + b
    assert c.requires_grad

    d = a + a
    assert not d.requires_grad


def test_gradient_chain():
    """Test the gradient computation for a chain of operations."""

    a = Tensor([2, 3, 4], dtype=np.float32, requires_grad=True)
    b = a * 2  # [4, 6, 8]
    c = b + 3  # [7, 9, 11]
    d = c**2  # [49, 81, 121]

    d.gradient()

    # The gradient should be 2 * c * (chain rule through all operations)
    expected_grad = 2 * c.data * 2
    assert_grad(a, expected_grad)


def test_complex_chain():
    """Test the gradient computation for a complex chain of operations."""

    a = Tensor([1, 2, 3], dtype=np.float32, requires_grad=True)
    b = Tensor([4, 5, 6], dtype=np.float32, requires_grad=True)

    c = a * b
    d = c + a
    e = d / b
    f = e**2

    f.gradient()

    grad_e = 2 * e.data  # df/de
    grad_d = grad_e * (1 / b.data)  # df/dd = grad_e * d/db
    grad_c = grad_d  # df/dc = df/dd (chain rule)
    grad_a = grad_d * b.data + grad_c  # df/da = grad_d * db/da + grad_c
    grad_b = grad_d * a.data - grad_e * d.data / (b.data**2)  # df/db

    assert_grad(e, grad_e)
    assert_grad(d, grad_d)
    assert_grad(c, grad_c)
    assert_grad(b, grad_b)
    assert_grad(a, grad_a)


def test_mixed_operations():
    """Test the gradient computation for a chain of mixed operations."""

    a = Tensor([2, 3, 4], dtype=np.float32, requires_grad=True)
    b = Tensor([1, 2, 3], dtype=np.float32, requires_grad=True)
    c = a + b  # [3, 5, 7]
    d = c * a  # [6, 15, 28]
    e = d - b  # [5, 13, 25]
    f = e / a  # [2.5, 4.3333, 6.25]

    f.gradient()

    grad_e = 1 / a.data  # df/de
    grad_d = grad_e  # df/dd = grad_e
    grad_c = grad_d * a.data  # df/dc = grad_d * dd/dc
    grad_a = (
        grad_c + grad_d * c.data - e.data / (a.data**2)
    )  # df/da = df/da + dd/da + dc/da
    grad_b = grad_c - grad_e  # df/db = dc/db + de/db

    assert_grad(e, grad_e)
    assert_grad(d, grad_d)
    assert_grad(c, grad_c)
    assert_grad(b, grad_b)
    assert_grad(a, grad_a)


def test_exponential_operations():
    """Test the gradient computation for the exponential operation"""
    a = Tensor([1, 2, 3], dtype=np.float32, requires_grad=True)
    b = Tensor([2, 3, 4], dtype=np.float32, requires_grad=True)
    c = a**b  # [1, 8, 81]
    d = op.exp(c)  # [e, e^8, e^81]

    d.gradient()

    grad_c = d.data  # dd/dc
    grad_b = grad_c * (a.data**b.data) * np.log(a.data)  # dd/db = grad_c * dc/db
    grad_a = grad_c * b.data * (a.data ** (b.data - 1))  # dd/da = grad_c * dc/da

    assert_grad(c, grad_c)
    assert_grad(b, grad_b)
    assert_grad(a, grad_a)
