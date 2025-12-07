# session7_polynomial_regression.py
# Practice script based on "Nonlinear and Polynomial Regression Models" session

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# -----------------------------
# 1. Generate Synthetic Data
# -----------------------------
# Quadratic relationship with noise
np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = 3 * X.squeeze()**2 - 2 * X.squeeze() + 5 + np.random.randn(50) * 10

# -----------------------------
# 2. Fit Polynomial Regression Models
# -----------------------------
def fit_polynomial_regression(X, y, degree):
    """Fit polynomial regression of given degree and return predictions + MSE."""
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y, y_pred)
    return y_pred, mse, model

# -----------------------------
# 3. Compare Different Degrees
# -----------------------------
degrees = [1, 2, 5]
plt.figure(figsize=(12, 6))

for i, d in enumerate(degrees, 1):
    y_pred, mse, model = fit_polynomial_regression(X, y, d)
    plt.subplot(1, len(degrees), i)
    plt.scatter(X, y, color="blue", label="Data")
    plt.plot(X, y_pred, color="red", label=f"Degree {d} Fit")
    plt.title(f"Degree {d} Polynomial\nMSE: {mse:.2f}")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# 4. Bias-Variance Demonstration
# -----------------------------
# Compare MSE across polynomial degrees
mse_scores = []
for d in range(1, 10):
    _, mse, _ = fit_polynomial_regression(X, y, d)
    mse_scores.append(mse)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), mse_scores, marker="o")
plt.title("Bias-Variance Tradeoff: MSE vs Polynomial Degree")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.show()

# -----------------------------
# 5. Nonlinear Regression Example (Exponential Fit)
# -----------------------------
# Example nonlinear data: exponential growth
X_nonlin = np.linspace(0, 3, 50).reshape(-1, 1)
y_nonlin = np.exp(X_nonlin).squeeze() + np.random.randn(50) * 0.5

# Fit polynomial regression as approximation
y_pred_nonlin, mse_nonlin, _ = fit_polynomial_regression(X_nonlin, y_nonlin, degree=3)

plt.figure(figsize=(8, 5))
plt.scatter(X_nonlin, y_nonlin, color="blue", label="Nonlinear Data")
plt.plot(X_nonlin, y_pred_nonlin, color="red", label="Polynomial Approximation (deg=3)")
plt.title(f"Nonlinear Regression Approximation\nMSE: {mse_nonlin:.2f}")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
