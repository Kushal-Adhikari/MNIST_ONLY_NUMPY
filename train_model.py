
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('mnist_test.csv')
df.head()
array = df.to_numpy()
m , n = array.shape
np.random.shuffle(array)

data_dev = array[0:1000].T
Y_dev = data_dev[0].astype(int)
X_dev = data_dev[1:n]/255

data_train = array[1000:m].T
Y_train = data_train[0].astype(int)
X_train = data_train[1:n]/255


def init_param(): # Initializing weights and biases
  W1 = np.random.randn(10, 784) * 0.01 # the 0.01 is for scaling  ---  10 neurons 784 inputs
  B1 = np.zeros((10,1))
  W2 = np.random.randn(10, 10) * 0.01 # the 0.01 is for scaling  ---  10 neurons 10 inputs
  B2 = np.zeros((10,1))
  return W1, B1, W2, B2

def ReLU(Z): # if greater than X return X if less than X return 0
  return np.maximum(Z , 0)

def softmax(Z): # Applying softmax function
  Z_shifted = Z - np.max(Z, axis=0, keepdims=True) # for numerical stability
  A = np.exp(Z_shifted) / np.sum(np.exp(Z_shifted), axis=0, keepdims=True)
  return A

def forward_prop(W1 , B1 , W2 , B2 , X):
  Z1 = W1.dot(X) + B1
  A1 = ReLU(Z1)
  Z2 = W2.dot(A1) + B2
  A2 =softmax(Z2)

  return Z1 , A1 , Z2 , A2

def one_hot(Y):
  one_hot_Y = np.zeros((Y.size , Y.max()+1)) # returns 1 for the highest value everything else is zero
  one_hot_Y[np.arange(Y.size), Y] = 1
  one_hot_Y = one_hot_Y.T
  return one_hot_Y

def deriv_ReLU(Z): #Because ReLU is not differentiable at 0 we define its derivative as 0 for Z<=0 and 1 for Z>0
  return (Z > 0).astype(float)

def back_prop(Z1 , A1 , Z2 , A2 , W1 , W2 , X , Y):  
  m = Y.size
  one_hot_Y = one_hot(Y)
  dZ2 = A2 - one_hot_Y
  dW2 = 1 / m * dZ2.dot(A1.T)
  dB2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
  dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
  dW1 = 1 / m * dZ1.dot(X.T)
  dB1 = 1 / m * np.sum(dZ1 , axis = 1, keepdims=True)
  return dW1 , dB1 , dW2 , dB2

def update_param(W1 , B1 , W2 , B2, dW1 , dB1 , dW2 , dB2 , alpha): # Updating the parameters
  W1 = W1 - alpha * dW1
  B1 = B1 - alpha * dB1
  W2 = W2 - alpha * dW2
  B2 = B2 - alpha * dB2
  return W1 , B1 , W2 , B2


def get_predictions(A2):
  return np.argmax(A2 , 0)

def get_accuracy(predictions , Y):
  print(predictions , Y)
  return np.sum(predictions == Y) / Y.size


def gradient_descent(X,Y, iterations , alpha):
  W1 , B1 , W2 , B2 = init_param() # Initialize parameters
  
  for i in range(iterations):
    Z1 , A1, Z2 , A2 = forward_prop(W1 , B1 , W2 , B2 , X)
    dW1 , dB1 , dW2 , dB2 = back_prop(Z1 , A1, Z2, A2 , W1 , W2 , X , Y)
    W1 , B1 , W2 , B2 = update_param(W1 , B1 , W2 , B2, dW1 , dB1 , dW2 , dB2 , alpha)
    if i % 50 == 0:
      loss = -np.mean(one_hot(Y) * np.log(A2 + 1e-8))
      print(f"Iteration {i}, Loss: {loss}, Accuracy: {get_accuracy(get_predictions(A2), Y)}")

    if i % 25 == 0:
      print("Iteration: ", i)
      print("Accuracy: " , get_accuracy(get_predictions(A2), Y))
    

    

  return W1 , B1 , W2 , B2

if __name__ == "__main__":
  W1 , B1 , W2 , B2 = gradient_descent(X_train , Y_train , 2000 , 0.1)
  np.savez("mnist_parameters.npz", W1=W1, B1=B1, W2=W2, B2=B2)


