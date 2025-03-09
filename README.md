# Neural Network Implementation

A simple implementation of a neural network from scratch in Python for classification problems. This project provides a basic neural network class that can be used with different datasets.

## Features

- Feed-forward neural network implementation
- Flexible architecture that can work with various datasets
- Automatic differentiation using autograd for gradient computation
- Data visualization capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/joraleoca/neural-network.git
cd neural-network
```

2. Create and activate a virtual environment:
```bash
# Create venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Or activate on Mac/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements
### Core Requirements
- Python 3.x
- Cupy - for numerical computations

### Optional Dependencies
- Matplotlib - for data visualization (only used for training debug)
- Pandas - if you want to work with DataFrame inputs

## Usage

Here's a basic example of how to use the neural network:

```python
from src import NeuralNetwork, Config
from src.tensor import Tensor
from src.config import FeedForwardConfig, TrainingConfig
from src.loss import CategoricalCrossentropy
from src.structure import Dense, LeakyRelu, Softmax

# Set the default device for all tensor operations, including those created internally by the neural network
Config.set_default_device("cuda")

# Prepare your data as numpy arrays
X_train = Tensor(...)  # Your training data
y_train = Tensor(...)  # Your training labels

X_test = Tensor(...)  # Your testing data
y_test = Tensor(...)  # Your testing labels

classes = ("class1", "class2", "class3")

config = FeedForwardConfig(
    network_structure=[
        Dense(X_train.shape[1]), 
        LeakyRelu(),
        Dense(64),
        LeakyRelu(),
        Dense(32),
        LeakyRelu(),
        Dense(len(classes)),
        Softmax(),
    ],
    classes=classes,
)

# Initialize the neural network
nn = NeuralNetwork(config)

# Train the model
nn.train(
    X_train,
    y_train,
    X_test,
    y_test,
    config=TrainingConfig(
        loss=CategoricalCrossentropy()
    ),
)

# Make prediction for a single data point
prediction = nn.forward_pass(X_test[0])
```

### Example with Pandas (Optional)
```python
import pandas as pd
import numpy as np
from src import NeuralNetwork

# If using pandas
data = pd.read_csv('your_dataset.csv')
X = data.drop('target', axis=1).to_numpy()  # Convert to numpy array
y = data['target'].to_numpy()  # Convert to numpy array
```

**See the usage example file for more details. You need to install the example dependencies to run the example:**

```bash
pip install -r requirements.example.txt
```

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements.

## License

This project is licensed under the GNU General Public License v3.0.
