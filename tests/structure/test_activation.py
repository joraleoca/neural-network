import pytest
import numpy as np

from tests.utils import assert_data, assert_grad
from src.tensor import Tensor
from src.structure import LeakyRelu, Relu, Sigmoid, Softmax, Tanh


@pytest.fixture()
def data() -> Tensor:
    return Tensor(np.array([-1.0, 0.0, 1.0, 2.0]))


@pytest.mark.parametrize(
    "operation, expected_func",
    [
        (LeakyRelu(alpha=0.01), lambda x: x * (x > 0) + x * 0.01 * (x <= 0)),
        (Relu(), lambda x: x * (x > 0)),
        (Sigmoid(), lambda x: 1 / (1 + np.exp(-x))),
        (
            Softmax(),
            lambda x: np.exp(x - x.max()) / np.sum(np.exp(x - x.max())),
        ),
        (Tanh(), lambda x: np.tanh(x)),
    ],
    ids=lambda x: x.__class__.__name__,
)
def test_activation_forward(operation, data, expected_func):
    output = operation(data)

    assert_data(output, expected_func(data))


@pytest.mark.parametrize(
    "operation, expected_grad_input",
    [
        (LeakyRelu(alpha=0.01), np.array([0.01, 0.01, 1, 1])),
        (Relu(), np.array([0, 0, 1, 1])),
        (Sigmoid(), np.array([0.19661193, 0.25, 0.19661193, 0.10499359])),
        (Tanh(), np.array([0.41997434, 1.0, 0.41997434, 0.07065082])),
    ],
    ids=lambda x: x.__class__.__name__,
)
def test_activation_backward(operation, data, expected_grad_input):
    data.requires_grad = True
    output = operation(data)

    output.backward()

    assert_grad(data, expected_grad_input)

    data.zero_grad()


@pytest.mark.parametrize("alpha", [-0.01, 0, -1.0])
def test_leaky_relu_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        LeakyRelu(alpha=alpha)


def test_softmax_forward(data):
    output = Softmax()(data)

    exp_x = np.exp(data.data - np.max(data.data))
    expected_output = exp_x / np.sum(exp_x)
    assert_data(output, expected_output)
