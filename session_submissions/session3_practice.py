import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate synthetic data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Define a simple linear regression model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(X.shape[1],))
])

# Compile the model with SGD optimizer and MSE loss
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss='mean_squared_error')

# Train the model
history = model.fit(X, y, epochs=200, verbose=0)

# Extract learned parameters
weights = model.layers[0].get_weights()
beta_1 = weights[0][0][0]  # slope
beta_0 = weights[1][0]     # intercept

print("Learned Parameters:")
print(f"Intercept (β0): {beta_0:.3f}")
print(f"Slope (β1): {beta_1:.3f}")

# Plot training loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', linewidth=2)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True, alpha=0.3)

# Plot data and fitted line
plt.subplot(1, 2, 2)
plt.scatter(X, y, color="blue", alpha=0.5, label="Data")
X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)
plt.plot(X_new, y_pred, color="red", linewidth=2, label="Fitted Line")
plt.title("Linear Regression Fit")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
