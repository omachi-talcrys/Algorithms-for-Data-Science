# AI Persona Feature Feedback Simulator

This project is a **Streamlit-based AI simulation tool** for generating persona-based feedback on product features. It uses TinyTroupe and OpenAI to simulate responses from custom or predefined personas.

## Features

- **Single Persona Feedback:** Generate AI feedback for a given feature using one persona.
- **Scenario Simulation:** Test features under predefined scenarios like onboarding, feature discovery, long-term usage, and abandonment risk.
- **Batch Persona Testing:** Run multiple personas at once and compare their feedback.
- **Analytics & Visualization:** Summarizes sentiment, themes, and persona behavior trends.
- **Export Reports:** Export feedback and analysis as CSV, PDF, or text files.

## Installation

**1. Clone the repository:**


git clone https://github.com/omachi-talcrys/Algorithms-for-Data-Science.git
cd Algorithms-for-Data-Science

**2. Install dependencies:**
pip install -r requirements.txt

Set your OpenAI API key:
setx OPENAI_API_KEY "YOUR_API_KEY"

**3. Usage**
Run the Streamlit app:
streamlit run app.py
- Select a predefined or custom persona.

- Enter a feature description.

- Simulate feedback or run scenarios.

- View AI-generated responses and analytics in the UI.

Notes
API Key: Replace "YOUR_API_KEY" in the environment variables. Do not commit your key to GitHub.

Timeout & Tokens: Large prompts may require higher MAX_TOKENS and longer TIMEOUT in the config.

Crashes: Some experimental features may cause the app to crash; these are known issues.
