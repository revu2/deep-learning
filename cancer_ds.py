# ================================
# IMPORT LIBRARIES
# ================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ================================
# LOAD & PREPROCESS DATASET
# ================================
df = pd.read_csv("breast-cancer.csv")   # Kaggle dataset
df = df.drop(columns=["id"])            # remove ID
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

X = df.drop(columns=["diagnosis"]).values   # 30 features
y = df["diagnosis"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ================================
# ACTIVATION FUNCTIONS
# ================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def step(z):
    return (z >= 0).astype(int)


# ================================
# 1) SIGMOID PERCEPTRON
# ================================
def train_sigmoid_perceptron(X, y, lr=0.01, epochs=1000):
    w = np.zeros((X.shape[1], 1))
    b = 0

    for _ in range(epochs):
        y_hat = sigmoid(np.dot(X, w) + b)
        w -= lr * np.dot(X.T, (y_hat - y)) / len(y)
        b -= lr * np.mean(y_hat - y)

    return w, b

def evaluate_sigmoid(X, y, w, b):
    y_pred = sigmoid(np.dot(X, w) + b)
    y_class = (y_pred >= 0.5).astype(int)
    acc = np.mean(y_class == y)
    error = np.mean((y_pred - y) ** 2)
    return acc, error


# ================================
# 2) THRESHOLD PERCEPTRON
# ================================
def train_threshold_perceptron(X, y, lr=0.01, epochs=1000):
    w = np.zeros((X.shape[1], 1))
    b = 0

    for _ in range(epochs):
        y_hat = step(np.dot(X, w) + b)
        w += lr * np.dot(X.T, (y - y_hat))
        b += lr * np.sum(y - y_hat)

    return w, b

def evaluate_threshold(X, y, w, b):
    y_pred = step(np.dot(X, w) + b)
    acc = np.mean(y_pred == y)
    error = 1 - acc
    return acc, error


# ================================
# 3) ONE HIDDEN LAYER NEURAL NETWORK
# ================================
def train_mlp(X, y, lr=0.1, epochs=5000):
    W1 = np.random.randn(X.shape[1], 12) * 0.01
    b1 = np.zeros((1, 12))
    W2 = np.random.randn(12, 1) * 0.01
    b2 = np.zeros((1, 1))

    for _ in range(epochs):
        # Forward pass
        A1 = sigmoid(np.dot(X, W1) + b1)
        A2 = sigmoid(np.dot(A1, W2) + b2)

        # Backpropagation
        dZ2 = A2 - y
        dW2 = np.dot(A1.T, dZ2) / len(y)
        db2 = np.mean(dZ2, axis=0)

        dZ1 = np.dot(dZ2, W2.T) * A1 * (1 - A1)
        dW1 = np.dot(X.T, dZ1) / len(y)
        db1 = np.mean(dZ1, axis=0)

        # Update weights
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    return W1, b1, W2, b2

def evaluate_mlp(X, y, W1, b1, W2, b2):
    A1 = sigmoid(np.dot(X, W1) + b1)
    A2 = sigmoid(np.dot(A1, W2) + b2)
    y_class = (A2 >= 0.5).astype(int)
    acc = np.mean(y_class == y)
    error = np.mean((A2 - y) ** 2)
    return acc, error


# ================================
# TRAIN & EVALUATE ALL MODELS
# ================================
w1, b1 = train_sigmoid_perceptron(X_train, y_train)
train_acc1, train_err1 = evaluate_sigmoid(X_train, y_train, w1, b1)
test_acc1, test_err1 = evaluate_sigmoid(X_test, y_test, w1, b1)

w2, b2 = train_threshold_perceptron(X_train, y_train)
train_acc2, train_err2 = evaluate_threshold(X_train, y_train, w2, b2)
test_acc2, test_err2 = evaluate_threshold(X_test, y_test, w2, b2)

W1, b1, W2, b2 = train_mlp(X_train, y_train)
train_acc3, train_err3 = evaluate_mlp(X_train, y_train, W1, b1, W2, b2)
test_acc3, test_err3 = evaluate_mlp(X_test, y_test, W1, b1, W2, b2)


# ================================
# DISPLAY OUTPUT
# ================================
print("\nModel\t\t\tTrain Acc\tTest Acc\tTrain Error\tTest Error")

print(f"Sigmoid Perceptron\t{train_acc1*100:.2f}%\t\t{test_acc1*100:.2f}%\t\t{train_err1*100:.2f}%\t\t{test_err1*100:.2f}%")

print(f"Threshold Perceptron\t{train_acc2*100:.2f}%\t\t{test_acc2*100:.2f}%\t\t{train_err2*100:.2f}%\t\t{test_err2*100:.2f}%")

print(f"1-Hidden Layer NN\t{train_acc3*100:.2f}%\t\t{test_acc3*100:.2f}%\t\t{train_err3*100:.2f}%\t\t{test_err3*100:.2f}%")
