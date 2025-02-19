import pytest

from src.structure import Dense
from src.activation import LeakyRelu, Sigmoid, Tanh
from src.initialization import LeCunNormal, HeNormal
from src.core import Tensor, op

class TestDense:
    def test_constructor_out_features(self):
        dense = Dense(10)

        assert dense.output_dim == 10, f"Output features should be 10. Got {dense.output_dim}."
        assert dense.activation_function is None, f"Activation function should be None. Got {dense.activation_function}."

    def test_constructor_features(self):
        dense = Dense((10, 100))

        assert dense.input_dim == 10, f"Input features should be 10. Got {dense.input_dim}."
        assert dense.output_dim == 100, f"Output features should be 100. Got {dense.output_dim}."
        assert dense.activation_function is None, f"Activation function should be None. Got {dense.activation_function}."

    @pytest.mark.parametrize(
        "features",
        [
            (0, 10),
            (10, 0),
            (0, 0),
            (-1, 10),
            (10, -1),
            (-1, -1),
            0,
            -1,
            -100,
            (1, 2, 3)
        ]
    )
    def test_exception_invalid_features(self, features):
        with pytest.raises(ValueError):
            Dense(features)

    @pytest.mark.parametrize(
        "activation_function",
        [
            # Choosed random activation functions
            LeakyRelu(),
            Sigmoid(),
            Tanh(),
        ]
    )
    def test_activation_function(self, activation_function):
        dense = Dense(10, activation_function=activation_function)

        assert dense.activation_function == activation_function, f"Activation function should be {activation_function}. Got {dense.activation_function}."

    @pytest.mark.parametrize(
        "initializer",
        [
            LeCunNormal(),
            HeNormal(),
        ]
    )
    def test_initializer(self, initializer):
        dense = Dense(10, weights_initializer=initializer)

        assert dense.initializer == initializer, f"Initializer should be {initializer}. Got {dense.initializer}."

    def test_weights_initializated(self):
        dense = Dense((10, 100), weights_initializer=LeCunNormal())

        assert hasattr(dense, "weights"), "Weights should be initialized."
        assert dense.weights.shape == (10, 100), f"Weights shape should be (10, 100). Got {dense.weights.shape}."
    
    def test_induced_input_dim_and_weight_initializate(self):
        dense = Dense(10, weights_initializer=LeCunNormal())

        assert dense.input_dim == -1, f"Input features should be -1. Got {dense.input_dim}."

        dense.forward(Tensor([[1, 2, 3]]))

        assert dense.input_dim == 3, f"Input features should be 3. Got {dense.input_dim}."
        assert hasattr(dense, "weights"), "Weights should be initialized."
        assert dense.weights.shape == (3, 10), f"Weights shape should be (3, 10). Got {dense.weights.shape}."

    def test_forward(self):
        dense = Dense(10, weights_initializer=LeCunNormal())

        data = Tensor([[1, 2, 3]])
        output = dense.forward(data)

        assert isinstance(output, Tensor), "Output should be a Tensor."
        assert output.shape == (1, 10), f"Output shape should be (1, 10). Got {output.shape}."
        assert hasattr(dense, "weights"), "Weights should be initialized."
        assert dense.weights.shape == (3, 10), f"Weights shape should be (3, 10). Got {dense.weights.shape}."
        assert hasattr(dense, "biases"), "Biases should be initialized."
        assert dense.biases.shape == (1, 10), f"Biases shape should be (1, 10). Got {dense.biases.shape}."
        assert output == (data @ dense.weights) + dense.biases, "Output should be the dot product of data and weights plus biases."

    def test_forward_activation_function(self):
        dense = Dense(10, activation_function=LeakyRelu(), weights_initializer=LeCunNormal())

        data = Tensor([[1, 2, 3]])
        output = dense.forward(data)

        assert isinstance(output, Tensor), "Output should be a Tensor."
        assert output.shape == (1, 10), f"Output shape should be (1, 10). Got {output.shape}."
        assert hasattr(dense, "weights"), "Weights should be initialized."
        assert dense.weights.shape == (3, 10), f"Weights shape should be (3, 10). Got {dense.weights.shape}."
        assert hasattr(dense, "biases"), "Biases should be initialized."
        assert dense.biases.shape == (1, 10), f"Biases shape should be (1, 10). Got {dense.biases.shape}."
        assert output == dense.activation_function((data @ dense.weights) + dense.biases), "Output should be the activation function of the forward output."

    def test_forward_dimensions_mismatch(self):
        dense = Dense((10, 10), weights_initializer=LeCunNormal())

        data = Tensor([[1, 2, 3, 4]])

        with pytest.raises(ValueError):
            dense.forward(data)

    def test_forward_dimensions_mismatch_induced(self):
        dense = Dense(10, weights_initializer=LeCunNormal())

        data = Tensor([[1, 2, 3, 4]])

        dense.forward(data)

        with pytest.raises(ValueError):
            dense.forward(Tensor([[1, 2, 3, 4, 5, 6]]))

    def test_rng(self):
        dense1 = Dense((10, 10), weights_initializer=LeCunNormal(), rng=42)
        dense2 = Dense((10, 10), weights_initializer=LeCunNormal(), rng=42)

        assert dense1.rng == dense2.rng, "RNG should be the same for both layers."
        assert dense1.weights == dense2.weights, "Weights should be the same for both layers."

        dense3 = Dense((10, 10), weights_initializer=LeCunNormal(), rng=43)

        assert dense1.rng != dense3.rng, "RNG should be different for both layers."
        assert dense1.weights != dense3.weights, "Weights should be different for both layers."

    def test_backwards(self):
        dense = Dense(10, weights_initializer=LeCunNormal())

        dense.requires_grad = True

        data = Tensor([[1, 2, 3]])
        output = dense.forward(data)

        output.backward()

        grad = op.zeros_like(output)
        grad.fill(1)
        assert data.T * grad == dense.weights.grad, "Grad input should be the dot product of grad and weights transposed." 
