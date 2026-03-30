import numpy as np

# Input (features) and Output (target)
X = np.array([1, 2, 3, 4, 5], dtype=float)  # feature
Y = np.array([2, 4, 6, 8, 10], dtype=float) # target

# Initialize parameters
theta0 = 0.0  # intercept
theta1 = 0.0  # slope
alpha = 0.01  # learning rate
epochs = 100  # number of iterations
batch_size = 2  # size of each mini-batch
m = len(X)

# Mini-Batch Gradient Descent
for _ in range(epochs):
    # Shuffle data each epoch
    indices = np.arange(m)
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]
    
    # Process mini-batches
    for start in range(0, m, batch_size):
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        Y_batch = Y_shuffled[start:end]
        
        # Predictions for mini-batch
        Y_pred = theta0 + theta1 * X_batch
        # Compute gradients
        d_theta0 = (1/len(X_batch)) * np.sum(Y_pred - Y_batch)
        d_theta1 = (1/len(X_batch)) * np.sum((Y_pred - Y_batch) * X_batch)
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
