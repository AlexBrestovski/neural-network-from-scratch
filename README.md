# Neural Network From Scratch

A feedforward neural network implemented from scratch using Python and NumPy, without deep-learning frameworks. The project explores the mathematical foundations of neural networks through forward propagation, backpropagation, and gradient descent.

## Overview

This project implements a neural network that learns the XOR classification problem. It demonstrates how a network can learn a nonlinear relationship from labeled examples by adjusting its weights and biases during training.

## Features

- Neural network implemented using NumPy
- ReLU activation function for the hidden layer
- Softmax activation for multiclass output
- Categorical cross-entropy loss
- Backpropagation using the chain rule
- Mini-free, full-batch gradient descent
- He-style weight initialization
- Training loss tracking and visualization
- Prediction and accuracy evaluation

## Architecture

The network contains three layers:

1. **Input layer:** 2 input features
2. **Hidden layer:** 8 neurons with ReLU activation
3. **Output layer:** 2 neurons with softmax activation

The network is trained using the four possible binary input combinations for XOR.

## Mathematical Foundations

### Forward propagation

The hidden layer calculates:

Z1 = X * W1 + b1

A1 = ReLU(Z1)

The output layer calculates:

Z2 = A1 * W2 + b2

A2 = softmax(Z2)

### Activation functions

ReLU:

Returns the maximum out of Z and 0

Softmax:

Return e^z / Sum(e^z)

### Loss function

Categorical cross-entropy measures the difference between the predicted probabilities and the correct one-hot labels:

Returns -Sum(Y * log(A2))

### Backpropagation

The output-layer gradient for softmax combined with cross-entropy simplifies to:

dZ2 = A2 - Y

The gradients for the output weights and biases are:

dW2 = (A1^T * dZ2) / m

dB2 = Sum(dZ2) / m

The error is then propagated into the hidden layer using the chain rule and the derivative of ReLU. The resulting gradients are used to update the network's parameters with gradient descent.

## Dataset

The XOR dataset contains four examples:

| Input | Expected output |
|---|---:|
| `[0, 0]` | 0 |
| `[0, 1]` | 1 |
| `[1, 0]` | 1 |
| `[1, 1]` | 0 |

The expected classes are represented using one-hot encoding during training.

## Installation

Requires Python 3 and pip.

Clone the repository:

```bash
git clone https://github.com/AlexBrestovski/neural-network-from-scratch.git
cd neural-network-from-scratch
```

Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

Run the XOR training experiment:

```bash
python train_xor.py
```

The program trains the network, prints the training loss, evaluates predictions against the expected classes, and displays the loss curve.

## Results

After running the experiment, document the results you actually obtain:

- Final training loss: add your measured value
- Training accuracy: add your measured value
- Predicted classes: add your program's output

Include a screenshot of your loss curve once you have verified that the network learns the XOR pattern.

## Learning Outcomes

This project helped me develop a deeper understanding of:

- Matrix operations in neural networks
- Activation functions and their derivatives
- The chain rule and backpropagation
- Cross-entropy loss and softmax
- Gradient descent and parameter updates
- Evaluating and visualizing model training

## Future Improvements

- Add support for multiple hidden layers
- Implement numerical gradient checking
- Experiment with different learning rates and hidden-layer sizes
- Add support for other small classification datasets
- Compare training behavior across different activation functions

## Technologies

- Python
- NumPy
- Matplotlib

**Note:** This project is an educational implementation intended to demonstrate the fundamentals of neural networks, rather than replace production deep-learning frameworks.
