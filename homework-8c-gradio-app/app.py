import gradio as gr
import openai
import json
from mnist_tools import mnist_logistic_regression, get_mnist_model_info

# -------------------------------
# OPENAI TOOL DEFINITIONS
# -------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mnist_logistic_regression",
            "description": "Train MNIST logistic regression with custom parameters",
            "parameters": {
                "type": "object",
                "properties": {
                    "epochs": {"type": "integer"},
                    "lr": {"type": "number"},
                    "batch_size": {"type": "integer"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_mnist_model_info",
            "description": "Get technical information about the MNIST model"
        }
    }
]

# -------------------------------
# CHAT FUNCTION
# -------------------------------
def chat(api_key, user_message, history):
    if not api_key:
        history.append(["System", "❌ Please enter your OpenAI API key."])
        return history

    openai.api_key = api_key

    # Prepare messages for OpenAI
    messages = [{"role": "system", "content": "You are an AI assistant with access to ML tools."}]
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": user_message})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        # Get the message content
        msg = response.choices[0].message
        msg_content = msg.content if hasattr(msg, "content") else ""

        # Check for tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_call = msg.tool_calls[0]
            tool_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")

            if tool_name == "mnist_logistic_regression":
                result = mnist_logistic_regression(
                    epochs=args.get("epochs", 2),
                    lr=args.get("lr", 0.1),
                    batch_size=args.get("batch_size", 64)
                )
            elif tool_name == "get_mnist_model_info":
                result = get_mnist_model_info()
            else:
                result = "Unknown tool."

            # Append the tool output as assistant response
            history.append([user_message, result])
            return history

        # Normal chat response
        history.append([user_message, msg_content])
        return history

    except Exception as e:
        history.append([user_message, f"❌ Error: {str(e)}"])
        return history

# -------------------------------
# GRADIO UI
# -------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 OpenAI MNIST Tool Chatbot")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔑 API Settings")
            api_key = gr.Textbox(label="OpenAI API Key", type="password")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot()
            user_input = gr.Textbox(label="Chat")
            send_btn = gr.Button("Send")

    send_btn.click(chat, inputs=[api_key, user_input, chatbot], outputs=chatbot)

demo.launch()
