# session6_model_selection.py
# Practice script based on "Model Selection and Related Concepts" session

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

# -----------------------------
# 1. Generate Synthetic Data
# -----------------------------
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

# -----------------------------
# 2. Error Measurement
# -----------------------------
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Residual Sum of Squares (RSS)
rss = np.sum((y - y_pred) ** 2)
print("Residual Sum of Squares (RSS):", rss)

# AIC and BIC (simplified calculation)
n = len(y)
p = X.shape[1]
mse = mean_squared_error(y, y_pred)
aic = n * np.log(mse) + 2 * p
bic = n * np.log(mse) + p * np.log(n)
print("AIC:", aic)
print("BIC:", bic)

# -----------------------------
# 3. Forward Stepwise Selection
# -----------------------------
forward_selector = SequentialFeatureSelector(model, direction='forward', n_features_to_select=5)
forward_selector.fit(X, y)
print("Forward Selected Features:", forward_selector.get_support())

# -----------------------------
# 4. Backward Stepwise Selection
# -----------------------------
backward_selector = SequentialFeatureSelector(model, direction='backward', n_features_to_select=5)
backward_selector.fit(X, y)
print("Backward Selected Features:", backward_selector.get_support())

# -----------------------------
# 5. Cp Statistic (simplified)
# -----------------------------
rss = np.sum((y - y_pred) ** 2)
sigma2 = mse  # approximate error variance
cp = rss / sigma2 - (n - 2 * p)
print("Cp Statistic:", cp)

# -----------------------------
# 6. Adjusted R²
# -----------------------------
r2 = model.score(X, y)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("R²:", r2)
print("Adjusted R²:", adj_r2)

# -----------------------------
# 7. Ridge and Lasso Regression
# -----------------------------
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
print("Ridge Coefficients:", ridge.coef_)

lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
print("Lasso Coefficients:", lasso.coef_)

# -----------------------------
# 8. ElasticNet Regression
# -----------------------------
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
print("ElasticNet Coefficients:", elastic_net.coef_)
