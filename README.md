# Neural Network Implementation

A simple implementation of a neural network from scratch in Python. This project provides a basic neural network class that can be used with different datasets.

## Features

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
To see how the library works, check out the **usage example folder**.
You need to install the example dependencies to run the example:

```bash
pip install -r requirements.example.txt
```

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements.

## License

This project is licensed under the GNU General Public License v3.0.
