import numpy as np
x1=np.array([0,0,1,1])
x2=np.array([0,1,0,1])

t=np.array([0,1,1,1])
m=len(t)
max_iter=10000
learning_rate=0.1
w1=np.random.uniform(-1,1)*0.5
w2=np.random.uniform(-1,1)*0.5

b=0
def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
    return z*(1-z)

for i in range(max_iter):
    db=0
    dw1=0
    dw2=0
    for j in range(m):
        z=w1*x1[j]+w2*x2[j]+b
        y_pred=sigmoid(z)
        error=t[j]-y_pred
        dpred_dz=sigmoid_derivative(y_pred)
        dw1 += x1[j]*error*dpred_dz
        dw2 += x2[j]*error*dpred_dz
        db += error*dpred_dz
    w1 += learning_rate*dw1
    w2 += learning_rate*dw2
    b += learning_rate*db
print("Trained weights and bias:")
print("w1:", w1)
print("w2:", w2)
print("b:", b)
print("Predictions after training:")
for j in range(m):
    z=w1*x1[j]+w2*x2[j]+b
    y_pred=sigmoid(z)
    print(f"Input: ({x1[j]}, {x2[j]}) => Predicted Output: {y_pred.round(0)}")
