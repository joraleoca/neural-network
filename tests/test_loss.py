import pytest
import numpy as np

from core import Tensor, constants as c

from utils import assert_grad, assert_data
import loss


@pytest.fixture()
def binary_cross_entropy():
    return loss.BinaryCrossentropy()


@pytest.fixture()
def categorical_cross_entropy():
    return loss.CategoricalCrossentropy()


class TestBinaryCrossentropy:
    @pytest.mark.parametrize(
        "expected, predicted, expected_loss",
        [
            (0, 0.5, -np.log(0.5)),
            (1, 0.5, -np.log(0.5)),
            (1, 0.9, -np.log(0.9)),
            (0, 0.1, -np.log(0.9)),
        ],
    )
    def test_binary_cross_entropy(
        self, expected, predicted, expected_loss, binary_cross_entropy
    ):
        """Test the forward computation for the binary cross-entropy loss."""
        loss = binary_cross_entropy(expected, predicted)

        assert_data(loss, expected_loss)

    @pytest.mark.parametrize(
        "expected, predicted, expected_grad",
        [
            (0, Tensor(0.5, requires_grad=True), np.array(2)),
            (1, Tensor(0.5, requires_grad=True), np.array(-2)),
            (1, Tensor(0.9, requires_grad=True), np.array(-1 / 0.9)),
            (0, Tensor(0.1, requires_grad=True), np.array(1 / 0.9)),
        ],
    )
    def test_binary_cross_entropy_backward(
        self, expected, predicted, expected_grad, binary_cross_entropy
    ):
        """Test the backward computation for the binary cross-entropy loss."""
        loss = binary_cross_entropy(expected, predicted)

        loss.backward()

        assert_grad(predicted, expected_grad)


class TestCategoricalCrossentropy:
    @pytest.mark.parametrize(
        "expected, predicted, expected_loss",
        [
            (
                Tensor([1, 0, 0]),
                Tensor([0.7, 0.2, 0.1]),
                -np.sum([1, 0, 0] * np.log([0.7, 0.2, 0.1])),
            ),
            (
                Tensor([0, 1, 0]),
                Tensor([0.7, 0.2, 0.1]),
                -np.sum([0, 1, 0] * np.log([0.7, 0.2, 0.1])),
            ),
            (
                Tensor([0, 0, 1]),
                Tensor([0.7, 0.2, 0.1]),
                -np.sum([0, 0, 1] * np.log([0.7, 0.2, 0.1])),
            ),
        ],
    )
    def test_categorical_cross_entropy(
        self, expected, predicted, expected_loss, categorical_cross_entropy
    ):
        """Test the forward computation for the categorical cross-entropy loss."""
        expected.data = np.clip(expected.data, c.EPSILON, 1 - c.EPSILON)

        loss = categorical_cross_entropy(expected, predicted)

        assert_data(loss, expected_loss)

    @pytest.mark.parametrize(
        "expected, predicted, expected_grad",
        [
            (
                Tensor([1, 0, 0]),
                Tensor([0.7, 0.2, 0.1], requires_grad=True),
                [-1 / 0.7, 0, 0],
            ),
            (
                Tensor([0, 1, 0]),
                Tensor([0.7, 0.2, 0.1], requires_grad=True),
                [0, -1 / 0.2, 0],
            ),
            (
                Tensor([0, 0, 1]),
                Tensor([0.7, 0.2, 0.1], requires_grad=True),
                [0, 0, -1 / 0.1],
            ),
        ],
    )
    def test_categorical_cross_entropy_backward(
        self, expected, predicted, expected_grad, categorical_cross_entropy
    ):
        """Test the backward computation for the categorical cross-entropy loss."""
        expected.data = np.clip(expected.data, c.EPSILON, 1 - c.EPSILON)

        loss = categorical_cross_entropy(expected, predicted)
        loss.backward()

        predicted.grad = predicted.grad.round(5)  # Due to clip operation

        assert_grad(predicted, expected_grad)
