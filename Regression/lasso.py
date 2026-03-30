
"""# **Lasso Regression**"""

import numpy as np

X = np.array([1,2,3,4])
y = np.array([2,4,6,8])
w = 0
lr = 0.01
lam = 0.1

for _ in range(1000):
    grad = np.mean((w*X-y)*X) + lam*np.sign(w)
    w -= lr*grad

pred = w*X
print("Weight:", w)
print("Error:", np.mean((y-pred)**2))
print("Accuracy:", (1-np.mean((y-pred)**2)/np.var(y))*100, "%")
