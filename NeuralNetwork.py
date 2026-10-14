import numpy as np
import matplotlib.pyplot as plt



class NeuralNetwork:
    def __init__(self, inputs_dim, hidden_dim, outputs_dim):
        self.W1 = np.random.randn(inputs_dim, hidden_dim) * np.sqrt(2 / inputs_dim)
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, outputs_dim) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros((1, outputs_dim))


    def relu(self,Z):
        return np.maximum(0, Z)


    def d_relu(self, Z):
        return (Z > 0).astype(float)


    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))

        return (exp_Z / np.sum(exp_Z, axis=1, keepdims=True))
    

    def forward_prop(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.relu(Z1)

        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.softmax(Z2)

        return Z1, A1, Z2, A2


    def get_loss(self, A2, Y):
        A2 = np.clip(A2, np.exp(-15), 1 - np.exp(-15))

        loss = -np.sum(Y * np.log(A2)) / Y.shape[0]

        return loss

    
    def backward_prop(self, Z1, A1, A2, Y, X):
        m = X.shape[0]

        dZ2 = A2 - Y
        dW2 = np.dot(A1.T, dZ2) / m 
        dB2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.d_relu(Z1)
        dW1 = np.dot(X.T, dZ1) / m
        dB1 = np.sum(dZ1, axis=0, keepdims=True) / m

        return dB1, dB2, dW1, dW2


    def gradient_descent(self, dB1, dB2, dW1, dW2, lr):
        self.W1 = self.W1 - lr * dW1
        self.b1 = self.b1 - lr * dB1

        self.W2 = self.W2 - lr * dW2
        self.b2 = self.b2 - lr * dB2


    def train(self, X, Y, epochs=1000, lr=0.1):
        losses = []
        for i in range(epochs):
            Z1, A1, Z2, A2 = self.forward_prop(X)

            loss = self.get_loss(A2, Y)
            losses.append(loss)

            dB1, dB2, dW1, dW2 = self.backward_prop(Z1, A1, A2, Y, X)

            self.gradient_descent(dB1, dB2, dW1, dW2, lr)

            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")

        return losses


    def predict(self, X):
        _, _, _, probabilities = self.forward_prop(X)
        
        return np.argmax(probabilities, axis=1)