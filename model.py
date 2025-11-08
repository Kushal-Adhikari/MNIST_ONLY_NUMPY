import numpy as np

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    A = np.exp(Z_shifted) / np.sum(np.exp(Z_shifted), axis=0, keepdims=True)
    return A

def forward_prop(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = softmax(Z2)
    return A2

def load_model():
    params = np.load("mnist_parameters.npz")
    W1 = params["W1"]
    B1 = params["B1"]
    W2 = params["W2"]
    B2 = params["B2"]
    print("W1 shape:", W1.shape)
    print("B1 shape:", B1.shape)
    print("W2 shape:", W2.shape)
    print("B2 shape:", B2.shape)
    return W1, B1, W2, B2

def predict(X):
    W1, B1, W2, B2 = load_model()
    A2 = forward_prop(W1, B1, W2, B2, X)
    predictions = np.argmax(A2, axis=0)
    return predictions
