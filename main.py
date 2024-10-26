import numpy as np
import pandas as pd

from neural_network import (
    NeuralNetwork,
    FunctionActivation,
    min_max_scaler,
    train_test_split,
)

import random


def main() -> None:
    classes = ("apple", "banana", "cherry")

    nn = NeuralNetwork(
        network_structure=[2, 5, 5, 3],
        classes=classes,
        hidden_activation=FunctionActivation.LEAKY_RELU,
        output_activation=FunctionActivation.SOFTMAX,
    )

    data = pd.read_csv(
        "fruit_features.csv", sep=",", usecols=["weight", "color_intensity"]
    ).to_numpy()

    labels = pd.read_csv("fruit_labels.csv").to_numpy(dtype=str).flatten()

    # Normalize data
    data = min_max_scaler(data, -1, 1)
    # Combine data and labels into a structured array
    combined = np.array([(d, label) for d, label in zip(data, labels)], dtype=object)

    # Split data into training and testing sets
    train, test = train_test_split(
        combined, train_size=1 / 8, test_size=1 / 16, random_state=0
    )

    nn.train(
        list(train),
        list(test),
        epochs=3000,
        debug=True,
    )

    # Test the trained network
    for _ in range(10):
        i = random.randint(0, len(data) - 1)
        result = nn.forward_pass(data[i])
        predicted_class = classes[np.argmax(result)]
        print(f"Input: {data[i]}, Predicted: {predicted_class}, Actual: {labels[i]}")

    print(f"weigths: {nn.weights}, biases {nn.biases}")


if __name__ == "__main__":
    main()
