# 🧠 OpenAI MNIST Tool-Calling Chatbot

This project is a Gradio-based AI chatbot application that integrates the OpenAI API with machine learning tool calling. The chatbot can have normal conversations and also trigger machine learning tools when prompted.

## 📌 Features

- Clean Gradio interface with sidebar API key input
- Real-time OpenAI-powered chat interface
- Tool-calling support
- PyTorch Logistic Regression training on the MNIST dataset
- Automatic dataset download and preprocessing
- Configurable training parameters (epochs, learning rate, batch size)
- Model saving to disk
- Model information retrieval tool

## 📁 Project Structure

```
mnist-ai-chatbot/
├── app.py
├── mnist_tools.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation

1. Clone the repository:

```
git clone <your-repo-url>
cd mnist-ai-chatbot
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## ▶️ Usage

Run the application:
```
python app.py
```

The Gradio web interface will open in your browser.

## Available Tools

The chatbot can call the following tools:

1.  ``` mnist_logistic_regression ```

Trains a logistic regression model on the MNIST dataset.

**Configurable parameters:**

-  ``` epochs ```
- ``` lr ``` (learning rate)
- ``` batch_size ```

2. ``` get_mnist_model_info ```
- Returns technical information about:
- Model architecture
- Dataset size
- Saved model status

## 💬 Example Prompts

You can try:
- "Train a logistic regression model on MNIST with 5 epochs"
- "Show me model architecture information"
- "What is MNIST?"

## Educational Purpose

This project demonstrates:
- LLM tool calling
- API integration
- A real-world ML training pipeline
- Frontend + backend integration using Gradio and PyTorch
