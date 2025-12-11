import numpy as np

# ---------- Load data from file ----------
# data.csv format (no header):
# x1, y1
# x2, y2
# ...
data = np.loadtxt("linear_regression/linearwl.py", delimiter=",")

X = data[:, 0]          # feature column
y = data[:, 1]          # target column

# ---------- Prepare design matrix ----------
# Add a column of ones for bias term
X_b = np.c_[np.ones((X.shape[0], 1)), X]   # shape (m, 2)

# ---------- Hyperparameters ----------
learning_rate = 0.01
n_iters = 1000

# ---------- Initialize parameters ----------
theta = np.zeros(2)     # [theta0, theta1]
m = len(X)              # number of samples

# ---------- Gradient Descent Loop ----------
for i in range(n_iters):
    # Predictions
    y_pred = X_b.dot(theta)           # shape (m,)
    
    # Mean Squared Error gradient
    gradients = (2 / m) * X_b.T.dot(y_pred - y)
    
    # Update parameters
    theta = theta - learning_rate * gradients

# ---------- Results ----------
print("Learned parameters (theta0, theta1):", theta)
print("Model: y ≈ {:.3f} + {:.3f} * x".format(theta[0], theta[1]))
