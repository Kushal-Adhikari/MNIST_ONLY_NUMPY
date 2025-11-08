import numpy as np
import matplotlib.pyplot as plt
from model import predict

# Load test data
data = np.loadtxt("mnist_test.csv", delimiter=",", skiprows=1).T
Y_test = data[0].astype(int)
X_test = data[1:] / 255.0  # Normalize

preds = predict(X_test)

# Number of samples to display
num_samples = 10

for i in range(num_samples):
    img = X_test[:, i].reshape(28, 28)  # reshape to image
    predicted = preds[i]
    actual = Y_test[i]

    plt.imshow(img, cmap="gray")
    plt.title(f"Predicted: {predicted}, Actual: {actual}")
    plt.axis("off")
    plt.show()
