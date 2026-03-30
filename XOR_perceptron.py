# XOR GATE WITH SIGMOID
import numpy as np
def sigmoid(z):
  return 1/(1+np.exp(-z))
def sigmoid_derivative(y):
  return y*(1-y)

x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])
t  = np.array([0, 1, 1, 0])

m = len(t)
lr = 0.1
epochs = 5500

w1, w2 = np.random.randn(), np.random.randn()   # neuron 1
w3, w4 = np.random.randn(), np.random.randn()   # neuron 2
b1, b2 = 0, 0

#output neuron

v1,v2=np.random.randn(),np.random.randn()
b3=0

for _ in range(epochs):
  for i in range(m):
    z1=w1*x1[i]+w2*x2[i] + b1
    z2=w3*x1[i]+w4*x2[i] + b2

    h1=sigmoid(z1)
    h2=sigmoid(z2)

    z3=v1*h1+v2*h2 + b3
    y=sigmoid(z3)

    error=t[i]-y

    delta_out=error* sigmoid_derivative(y)
    delta_h1 = delta_out * v1 * sigmoid_derivative(h1)
    delta_h2 = delta_out * v2 * sigmoid_derivative(h2)

    v1 += lr * delta_out * h1
    v2 += lr * delta_out * h2
    b3 += lr * delta_out

    w1 += lr * delta_h1 * x1[i]
    w2 += lr * delta_h1 * x2[i]
    b1 += lr * delta_h1

    w3 += lr * delta_h2 * x1[i]
    w4 += lr * delta_h2 * x2[i]
    b2 += lr * delta_h2


z1 = w1*x1 + w2*x2 + b1
z2 = w3*x1 + w4*x2 + b2

h1 = sigmoid(z1)
h2 = sigmoid(z2)

z3 = v1*h1 + v2*h2 + b3
pred = sigmoid(z3)

print("Predictions:", pred.round())
