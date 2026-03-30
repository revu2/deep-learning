"""# **Ridge Regression**"""

import numpy as np

X = np.array([[1],[2],[3],[4]])
y = np.array([2,4,6,8])
lam = 0.1

w = np.linalg.inv(X.T@X + lam*np.eye(1)) @ X.T @ y
pred = X@w

print("Weight:", w)
print("Error:", np.mean((y-pred)**2))
print("Accuracy:", (1-np.mean((y-pred)**2)/np.var(y))*100, "%")
