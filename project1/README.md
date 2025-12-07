# Credibility Scoring Proof-of-Concept

## Project Overview

This project implements a Credibility Scoring System for online sources (articles, blogs, academic papers, government reports, etc.) using a hybrid approach. In today's information-rich environment, distinguishing credible from unreliable information is critical. This system provides a numerical credibility score along with a clear explanation to help users assess the reliability of any reference.

The system is designed as a proof-of-concept and can be integrated with chatbots or other applications requiring real-time source evaluation.

---

## Key Features

1. Hybrid Scoring Approach
   - Rule-Based: Evaluates source credibility using domain authority, citation presence, and content heuristics.
   - Machine Learning: Uses a simple Logistic Regression model with TF-IDF features to assess textual credibility.
   - Final Score: Combines rule-based and ML scores to generate a balanced credibility score between 0 and 1.

2. Robust Error Handling
   - Handles invalid URLs, inaccessible pages, and exceptions gracefully.
   - Provides meaningful explanations for score results.

3. JSON Output
   - Returns structured information:
   ```json
   {
  "score": 0.90,
  "explanation": "Domain suggests credible source. ML model prediction: 0.85"}
4. Extensible & Production-Ready
    - Modular Python function ready to be integrated with chatbots or web applications.
    - Can be extended with larger ML models or additional rule-based checks.

## How the ML Model Works
The ML component is a proof-of-concept classifier:

**Model**: Logistic Regression

**Features**: TF-IDF vectorization of webpage text

**Training Data**: A small set of example texts labeled as credible or not credible

The ML prediction represents the probability that the content is credible. This probability is combined with the rule-based score to calculate the final hybrid credibility score.

Note: In a production system, the model would be trained on a large, labeled dataset covering multiple domains and publication types.

Installation
Clone this repository:

git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
**Navigate to the project folder:**

cd YOUR_REPO/credibility_scoring
Install required packages:

pip install -r requirements.txt
Example requirements.txt:
requests
beautifulsoup4
validators
scikit-learn


from credibility_checker import evaluate_credibility

url = "https://www.nature.com/articles/s41586-020-2649-2"
result = evaluate_credibility(url)
print(result)
Sample Output:

json
{
  "score": 0.92,
  "explanation": "Domain suggests credible source. Presence of citations detected. ML model prediction: 0.95"
}

## Live Deployment

This project is deployed as a Hugging Face Space for real-time testing and demonstration. You can access the interactive app here:

**Credibility Scoring Live Demo** https://omachi-hugging-credibility-checker.hf.space/?logs=container&__theme=system&deep_link=tT4IdSzJy_U

Enter any valid URL to see the credibility score and explanation instantly.

Useful for integrating with chatbots, web apps, or for quick source evaluation.

## Next Steps & Improvements

Train the ML model on a large, labeled dataset for better accuracy.

Add additional rule-based checks for author credentials, publication date, and peer-review indicators.

Integrate with a chatbot or web interface for real-time source assessment.

Deploy model to Hugging Face or another platform for production-ready API access.


Chioma Madu / C.M.
Pace University 
Date: December 2025






