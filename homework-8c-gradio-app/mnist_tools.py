import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os

MODEL_PATH = "mnist_logreg.pt"


def mnist_logistic_regression(epochs=2, lr=0.1, batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    model = nn.Linear(28 * 28, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        for images, labels in train_loader:
            images = images.view(-1, 28 * 28)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), MODEL_PATH)

    return f"""
✅ MNIST Logistic Regression Training Complete

Epochs: {epochs}
Learning Rate: {lr}
Batch Size: {batch_size}
Model saved to: {MODEL_PATH}
"""


def get_mnist_model_info():
    info = """
📌 MNIST Logistic Regression Model Info

Architecture:
- Input: 784 features (28 x 28 pixels)
- Output: 10 classes
- Model type: Single-layer Linear (Logistic Regression)

Dataset:
- 60,000 training images
- 10,000 test images

Saved Model File:
- mnist_logreg.pt
"""

    if not os.path.exists(MODEL_PATH):
        info += "\n⚠️ Model has not been trained yet."

    return info
