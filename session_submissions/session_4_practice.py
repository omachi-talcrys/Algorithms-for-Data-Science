# logistic_regression_practice.py
# Practice script based on "Introduction to Classification and Logistic Regression" session

import numpy as np

# -----------------------------
# TensorFlow Implementation
# -----------------------------
import tensorflow as tf

def sigmoid_tf(x):
    """Sigmoid activation function."""
    return 1 / (1 + tf.exp(-x))

def linear_regression_tf(X, W, b):
    """Linear combination of inputs and weights."""
    return tf.linalg.matvec(X, W) + b

def logistic_regression_tf(X, W, b):
    """Logistic regression model with sigmoid activation."""
    z = linear_regression_tf(X, W, b)
    return sigmoid_tf(z)

def calc_bce_loss_tf(y, y_hat):
    """Binary cross-entropy loss."""
    epsilon = 1e-7
    y_hat = tf.clip_by_value(y_hat, epsilon, 1 - epsilon)
    return -tf.reduce_mean(y * tf.math.log(y_hat) + (1 - y) * tf.math.log(1 - y_hat))

def gradient_descent_tf(X, y, W, b, lr):
    """Gradient descent update for TensorFlow."""
    with tf.GradientTape() as tape:
        y_hat = logistic_regression_tf(X, W, b)
        loss = calc_bce_loss_tf(y, y_hat)
    gradients = tape.gradient(loss, [W, b])
    W.assign_sub(lr * gradients[0])
    b.assign_sub(lr * gradients[1])
    return W, b, loss

def predict_tf(X, W, b, threshold=0.5):
    """Predict labels using logistic regression."""
    y_hat = logistic_regression_tf(X, W, b)
    return tf.cast(y_hat >= threshold, dtype=tf.int32)

# -----------------------------
# PyTorch Implementation
# -----------------------------
import torch
import torch.nn as nn

class LogisticRegressionTorch(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegressionTorch, self).__init__()
        self.linear = nn.Linear(n_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    # Toy dataset
    X_np = np.array([[0.5, 1.2], [1.0, -0.5], [-1.2, 2.0], [0.3, 0.8]], dtype=np.float32)
    y_np = np.array([1, 0, 1, 0], dtype=np.float32)

    # TensorFlow training
    X_tf = tf.constant(X_np)
    y_tf = tf.constant(y_np)
    W_tf = tf.Variable(tf.zeros(X_np.shape[1]))
    b_tf = tf.Variable(0.0)

    print("\n--- TensorFlow Training ---")
    for epoch in range(5):
        W_tf, b_tf, loss_tf = gradient_descent_tf(X_tf, y_tf, W_tf, b_tf, lr=0.1)
        print(f"Epoch {epoch+1}, Loss: {loss_tf.numpy():.4f}")

    preds_tf = predict_tf(X_tf, W_tf, b_tf)
    print("Predicted labels (TF):", preds_tf.numpy())

    # PyTorch training
    X_torch = torch.tensor(X_np)
    y_torch = torch.tensor(y_np).view(-1, 1)
    model = LogisticRegressionTorch(n_features=2)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    print("\n--- PyTorch Training ---")
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X_torch)
        loss = criterion(outputs, y_torch)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    preds_torch = (outputs.detach().numpy() >= 0.5).astype(int)
    print("Predicted labels (Torch):", preds_torch.flatten())
