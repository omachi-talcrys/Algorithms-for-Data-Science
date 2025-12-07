# session5_sampling_bootstrap_cv.py
# Practice script based on "Sampling, Bootstrap, and Cross Validation" session

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

# -----------------------------
# 1. Sampling Examples
# -----------------------------
data = np.arange(1, 21)  # dataset of 20 points

# Random Sampling
random_sample = np.random.choice(data, size=5, replace=False)
print("Random Sample:", random_sample)

# Stratified Sampling (simulate two groups)
group_A = data[:10]
group_B = data[10:]
stratified_sample = np.concatenate([
    np.random.choice(group_A, size=3, replace=False),
    np.random.choice(group_B, size=2, replace=False)
])
print("Stratified Sample:", stratified_sample)

# Systematic Sampling (every 3rd element)
systematic_sample = data[::3]
print("Systematic Sample:", systematic_sample)

# -----------------------------
# 2. Bootstrap Example
# -----------------------------
original_data = [5, 10, 15, 20, 25]
bootstrap_sample = np.random.choice(original_data, size=len(original_data), replace=True)
print("Bootstrap Sample:", bootstrap_sample)

# Estimate mean variability with bootstrap
bootstrap_means = []
for _ in range(1000):
    sample = np.random.choice(original_data, size=len(original_data), replace=True)
    bootstrap_means.append(np.mean(sample))
print("Bootstrap Mean Estimate:", np.mean(bootstrap_means))
print("Bootstrap Std Dev:", np.std(bootstrap_means))

# -----------------------------
# 3. Cross Validation with Scikit-Learn
# -----------------------------
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
model = LinearRegression()

kf = KFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=kf)
print("Cross Validation Scores:", scores)
print("Mean Score:", scores.mean())

# -----------------------------
# 4. Cross Validation From Scratch
# -----------------------------
np.random.seed(42)
X = np.random.rand(100, 1)
y = 3 * X[:, 0] + np.random.randn(100) * 0.1

k = 5
fold_size = len(X) // k
mse_scores = []

for i in range(k):
    val_start = i * fold_size
    val_end = val_start + fold_size

    X_val = X[val_start:val_end]
    y_val = y[val_start:val_end]

    X_train = np.concatenate([X[:val_start], X[val_end:]], axis=0)
    y_train = np.concatenate([y[:val_start], y[val_end:]], axis=0)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mse_scores.append(mse)

print("Cross Validation MSEs:", mse_scores)
print("Mean MSE:", np.mean(mse_scores))

# -----------------------------
# 5. Variance and Error Measurement
# -----------------------------
scores = [0.85, 0.88, 0.84, 0.87, 0.86]
variance = np.var(scores)
print("Variance of CV Scores:", variance)

y_true = [1.5, 2.0, 1.8]
y_pred = [1.4, 2.1, 1.7]
error = mean_squared_error(y_true, y_pred)
print("Mean Squared Error:", error)
