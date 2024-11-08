# Neural Network Implementation

A simple implementation of a neural network from scratch in Python for classification problems. This project provides a basic neural network class that can be used with different datasets.

## Features

- Feed-forward neural network implementation
- Flexible architecture that can work with various datasets
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
- NumPy - for numerical computations
- Matplotlib - for data visualization

### Optional Dependencies
- Pandas - if you want to work with DataFrame inputs

## Usage

Here's a basic example of how to use the neural network:

```python
import numpy as np
from neural_network import NeuralNetwork

# Prepare your data as numpy arrays
X_train = np.array(...)  # Your training data
y_train = np.array(...)  # Your training labels
train_data = np.array([(d, label[0]) for d, label in zip(X_train, y_train)], dtype=object)

X_test = np.array(...)  # Your testing data
y_test = np.array(...)  # Your testing labels
test_data = np.array([(d, label[0]) for d, label in zip(X_test, y_test)], dtype=object)

# Initialize the neural network
nn = NeuralNetwork()

# Train the model
nn.train(list(train_data), list(test_data))

# Make predictions
predictions = nn.predict(X_test)
```

### Example with Pandas (Optional)
```python
import pandas as pd
import numpy as np
from neural_network import NeuralNetwork

# If using pandas
data = pd.read_csv('your_dataset.csv')
X = data.drop('target', axis=1).to_numpy()  # Convert to numpy array
y = data['target'].to_numpy()  # Convert to numpy array
```

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements.

## License

This project is licensed under the GNU General Public License v3.0.
