import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)

def linear(x):
    return x

def binary(x):
    return np.where(x >= 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def bipolar_sigmoid(x):
    return (2 / (1 + np.exp(-x))) - 1

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)

plt.figure(figsize=(12, 12))

plt.subplot(4, 2, 1)
plt.plot(x, linear(x))
plt.title("Linear")
plt.grid()

plt.subplot(4, 2, 2)
plt.plot(x, binary(x))
plt.title("Binary Step")
plt.grid()

plt.subplot(4, 2, 3)
plt.plot(x, sigmoid(x))
plt.title("Sigmoid")
plt.grid()

plt.subplot(4, 2, 4)
plt.plot(x, bipolar_sigmoid(x))
plt.title("Bipolar Sigmoid")
plt.grid()

plt.subplot(4, 2, 5)
plt.plot(x, tanh(x))
plt.title("Tanh")
plt.grid()

plt.subplot(4, 2, 6)
plt.plot(x, relu(x))
plt.title("ReLU")
plt.grid()

plt.subplot(4, 2, 7)
plt.plot(x, leaky_relu(x))
plt.title("Leaky ReLU")
plt.grid()

plt.tight_layout()
plt.show()