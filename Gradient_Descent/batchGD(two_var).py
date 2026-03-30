import numpy as np

# Input (features) and Output (target)
X = np.array([1, 2, 3, 4, 5], dtype=float)  # feature
Y = np.array([2, 4, 6, 8, 10], dtype=float) # target

# Initialize parameters
theta0 = 0.0  # intercept
theta1 = 0.0  # slope
alpha = 0.01  # learning rate
epochs = 1000 # number of iterations
m = len(X)    # number of samples

# Batch Gradient Descent
for _ in range(epochs):
    # Predictions
    Y_pred = theta0 + theta1 * X
    # Compute gradients
    d_theta0 = (1/m) * np.sum(Y_pred - Y)
    d_theta1 = (1/m) * np.sum((Y_pred - Y) * X)
    # Update parameters
    theta0 -= alpha * d_theta0
    theta1 -= alpha * d_theta1

# Final model
print("Trained theta0 (intercept):", theta0)
print("Trained theta1 (slope):", theta1)

# Testing prediction
test_X = 6
pred_Y = theta0 + theta1 * test_X
print(f"Prediction for X={test_X} is Y={pred_Y}")
