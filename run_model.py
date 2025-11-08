import numpy as np
from model import predict

# Load test data
test_data = np.loadtxt("mnist_test.csv", delimiter=",", skiprows=1).T
Y_test = test_data[0].astype(int)
X_test = test_data[1:] / 255.0   # shape (784, m)

# Run the trained model
preds = predict(X_test)

accuracy = np.mean(preds == Y_test)
print("Accuracy:", accuracy)
