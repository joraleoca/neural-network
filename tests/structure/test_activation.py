import pytest
import numpy as np

from tests.utils import assert_data, assert_grad
from src.tensor import Tensor
from src.structure import LeakyRelu, Relu, Sigmoid, Softmax, Tanh, layer_from_name


class TestActivation:
    @pytest.fixture(scope="class")
    def data(self) -> Tensor:
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
    def test_activation_forward(self, operation, data, expected_func):
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
    def test_activation_backward(self, operation, data, expected_grad_input):
        data.requires_grad = True
        output = operation(data)

        output.backward()

        assert_grad(data, expected_grad_input)

        data.zero_grad()

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
        assert layer_from_name(name) == class_type

    @pytest.mark.parametrize("invalid_name", ["activation_test", "relu", "", "Unknown"])
    def test_activation_from_name_error(self, invalid_name):
        with pytest.raises(ValueError):
            layer_from_name(invalid_name)
