# credibility_checker.py
"""
Credibility Scoring Proof-of-Concept
------------------------------------
This module provides a function to evaluate the credibility of web sources (URLs)
using a hybrid approach: rule-based heuristics + a simple ML model.

The ML model is a placeholder Logistic Regression trained on a few example texts.
In a production system, you would train on a large dataset of labeled credible vs.
non-credible articles. The ML component predicts textual credibility, which is then
combined with domain/citation heuristics for a hybrid score.
"""

import requests
from bs4 import BeautifulSoup
import re
import json
import validators

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --------------------------
# Credibility Scoring Function
# --------------------------
def evaluate_credibility(url):
    """
    Evaluate the credibility of a given URL.

    Args:
        url (str): The URL of the reference to evaluate.

    Returns:
        dict: JSON object containing:
              - score (float): credibility score between 0 and 1
              - explanation (str): rationale for the score
    """

    # Initialize default response
    result = {"score": 0.0, "explanation": ""}

    # --------------------------
    # Validate URL
    # --------------------------
    # Ensure input is a valid URL format; if not, return an error explanation
    if not validators.url(url):
        result["explanation"] = "Invalid URL provided."
        return result

    try:
        # Fetch webpage content with a short timeout for real-time use
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            result["explanation"] = "URL not accessible or returned non-200 status."
            return result

        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)

        # --------------------------
        # Rule-Based Scoring
        # --------------------------
        # Baseline credibility score
        score = 0.5
        explanation = ""

        # Rule: High-authority domains
        credible_domains = [".edu", ".gov", "nature.com", "sciencedirect.com", "wikipedia.org"]
        if any(domain in url for domain in credible_domains):
            score += 0.3
            explanation += "Domain suggests credible source."

        # Rule: Presence of citations or references
        if re.search(r'\b(references|doi|citations)\b', text, re.I):
            score += 0.1
            explanation += " Presence of citations detected."

        # Rule: Penalty for excessive scripts (heuristic for low-quality sites)
        if len(soup.find_all("script")) > 50:
            score -= 0.1
            explanation += " Many scripts found; may indicate low-quality site."

        # Clamp rule-based score between 0 and 1
        score = max(0.0, min(1.0, score))

        # --------------------------
        # ML-Based Scoring
        # --------------------------
        # For simplicity, we use a placeholder Logistic Regression model
        # trained on a few example texts. In production, a large labeled dataset
        # would allow accurate credibility predictions.
        fake_training_data = [
            "This is a peer-reviewed article from a science journal.",
            "This blog post is opinionated and has no references.",
            "Official government report on health.",
            "Random forum post with unverified information."
        ]
        fake_labels = [1, 0, 1, 0]  # 1 = credible, 0 = not credible

        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(fake_training_data)
        model = LogisticRegression()
        model.fit(X_train, fake_labels)

        # Transform the webpage text and predict credibility probability
        X_input = vectorizer.transform([text])
        ml_pred = model.predict_proba(X_input)[0][1]  # probability of credible

        # --------------------------
        # Hybrid Score Calculation
        # --------------------------
        # Combine rule-based score with ML prediction for final credibility
        hybrid_score = (score + ml_pred) / 2
        hybrid_score = max(0.0, min(1.0, hybrid_score))

        # Prepare final JSON result
        result["score"] = round(float(hybrid_score), 2)
        result["explanation"] = explanation + f" ML model prediction: {ml_pred:.2f}"

    except Exception as e:
        result["explanation"] = f"Error processing URL: {str(e)}"

    return result

# --------------------------
# Example Usage / Test Cases
# --------------------------
if __name__ == "__main__":
    test_urls = [
        "https://www.nature.com/articles/s41586-020-2649-2",
        "https://en.wikipedia.org/wiki/OpenAI",
        "https://some-random-blog.com/opinion",
        "invalid-url"
    ]

    for url in test_urls:
        print(json.dumps(evaluate_credibility(url), indent=2))
