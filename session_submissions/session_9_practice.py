# session9_svm.py
# Practice script based on "Support Vector Machine" session

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.svm import SVR

# -----------------------------
# 1. SVM Classification with scikit-learn
# -----------------------------
# Generate synthetic binary classification dataset
X, y = make_classification(n_samples=100, n_features=2, random_state=42)

# Create an SVM classifier with a linear kernel
classifier = svm.SVC(kernel='linear')
classifier.fit(X, y)

# Predict a new sample
sample = [[0.5, 0.5]]
prediction = classifier.predict(sample)
print("SVM Classifier Predicted class:", prediction)

# Plot decision boundary
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=30)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = classifier.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
           alpha=0.5, linestyles=['--', '-', '--'])
plt.title("SVM Classification Decision Boundary")
plt.show()

# -----------------------------
# 2. SVM Regression (SVR) with scikit-learn
# -----------------------------
# Generate synthetic regression dataset
X_reg = np.sort(5 * np.random.rand(40, 1), axis=0)
y_reg = np.sin(X_reg).ravel() + np.random.normal(0, 0.1, X_reg.shape[0])

# Fit regression model
regressor = SVR(kernel='rbf', C=1e3, gamma=0.1)
regressor.fit(X_reg, y_reg)

# Make prediction
sample_reg = [[1.5]]
predicted_value = regressor.predict(sample_reg)
print("SVR Predicted value for 1.5:", predicted_value)

# Plot regression fit
plt.figure(figsize=(6, 5))
plt.scatter(X_reg, y_reg, color='blue', label="Data")
plt.plot(X_reg, regressor.predict(X_reg), color='red', label="SVR Fit")
plt.title("Support Vector Regression Example")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# -----------------------------
# 3. Simple Linear SVM from Scratch
# -----------------------------
class SimpleSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        m, n = X.shape
        y_ = np.where(y <= 0, -1, 1)  # ensure labels are -1 or 1
        self.w = np.zeros(n)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    db = y_[idx]
                
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
    
    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)

# Example usage of scratch SVM
X_scratch = np.array([[1, 2], [2, 3], [3, 3]])
Y_scratch = np.array([-1, -1, 1])

svm_scratch = SimpleSVM()
svm_scratch.fit(X_scratch, Y_scratch)
predictions_scratch = svm_scratch.predict(X_scratch)
print("Scratch SVM Predicted labels:", predictions_scratch)
