import pytest
import numpy as np

from utils import assert_data, assert_grad
from src.core import Tensor
from src.activation import LeakyRelu, Relu, Sigmoid, Softmax, Tanh
from src.activation import activation_from_name


class TestActivation:
    @pytest.mark.parametrize(
        "operation, expected_output",
        [
            (LeakyRelu(alpha=0.01), np.array([-0.01, 0.0, 1.0, 2.0])),
            (Relu(), np.array([0.0, 0.0, 1.0, 2.0])),
            (Sigmoid(), 1 / (1 + np.exp(-np.array([-1.0, 0.0, 1.0, 2.0])))),
            (Tanh(), np.tanh(np.array([-1.0, 0.0, 1.0, 2.0]))),
        ],
    )
    def test_activation_forward(self, operation, expected_output):
        x = Tensor(np.array([-1.0, 0.0, 1.0, 2.0]))

        output = operation(x)

        assert_data(output, expected_output)


    @pytest.mark.parametrize(
        "operation, expected_grad_input",
        [
            (LeakyRelu(alpha=0.01), np.array([0.01, 0.01, 1, 1])),
            (Relu(), np.array([0, 0, 1, 1])),
            (Sigmoid(), np.array([0.19661193, 0.25, 0.19661193, 0.10499359])),
            (Tanh(), np.array([0.41997434, 1.0, 0.41997434, 0.07065082])),
        ],
    )
    def test_activation_backward(self, operation, expected_grad_input):
        x = Tensor(np.array([-1.0, 0.0, 1.0, 2.0]), requires_grad=True)

        output = operation(x)

        output.backward()

        assert_grad(x, expected_grad_input)

    @pytest.mark.parametrize("alpha", [-0.01, 0, -1.0])
    def test_leaky_relu_invalid_alpha(self, alpha):
        with pytest.raises(ValueError):
            LeakyRelu(alpha=alpha)


class TestSoftmax:
    @pytest.fixture()
    def softmax(self):
        return Softmax()

    def test_softmax_forward(self, softmax):
        x = Tensor(np.array([1.0, 2.0, 3.0]))

        output = softmax(x)

        exp_x = np.exp(x.data - np.max(x.data))
        expected_output = exp_x / np.sum(exp_x)
        assert_data(output, expected_output)

    # Softmax backwards is not implemented


class TestActivationUtils:
    @pytest.mark.parametrize(
        "name, class_type",
        [
            (Relu.__name__, Relu),
            (LeakyRelu.__name__, LeakyRelu),
            (Sigmoid.__name__, Sigmoid),
            (Tanh.__name__, Tanh),
            (Softmax.__name__, Softmax),
        ],
    )
    def test_activation_from_name(self, name, class_type):
        assert activation_from_name(name) == class_type

    @pytest.mark.parametrize("invalid_name", [
        "activation_test",
        "relu",
        "",
        "Unknown"
    ])
    def test_activation_from_name_error(self, invalid_name):
        with pytest.raises(ValueError):
            activation_from_name(invalid_name)