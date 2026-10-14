from NeuralNetwork import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt


X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)

Y = np.array([
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0]
    ], dtype=float)

np.random.seed(42)

nn = NeuralNetwork(inputs_dim=2, hidden_dim=8, outputs_dim=2)

losses = nn.train(X, Y, epochs=5000, lr=0.1)

predictions = nn.predict(X)
actual = np.argmax(Y, axis=1)

print("Predicted classes:", predictions)
print("Actual classes:   ", actual)
print("Accuracy:", np.mean(predictions == actual))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("Training Loss")
plt.show()