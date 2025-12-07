import pandas as pd
import plotly.express as px
from textblob import TextBlob

class FeedbackAnalyzer:
    def __init__(self):
        # Store all feedback by persona
        self.feedback_data = {}

    def add_feedback(self, persona_name, feedback):
        """Add feedback from a persona."""
        if persona_name not in self.feedback_data:
            self.feedback_data[persona_name] = []
        self.feedback_data[persona_name].append(feedback)

    def _sentiment_score(self, text):
        """Compute sentiment using TextBlob."""
        if not text or not isinstance(text, str):
            return 0
        return TextBlob(text).sentiment.polarity

    def aggregate_feedback(self):
        """
        Convert stored feedback into a DataFrame.
        Returns a DataFrame with persona, feedback, and sentiment.
        """
        rows = []
        for persona, feedback_list in self.feedback_data.items():
            for feedback in feedback_list:
                score = self._sentiment_score(feedback)
                rows.append({
                    "persona": persona,
                    "feedback": feedback,
                    "sentiment": score
                })

        df = pd.DataFrame(rows)
        return df

    def compute_consensus(self, df):
        """Determine whether personas generally agree or disagree."""
        avg_sentiment = df["sentiment"].mean()

        if avg_sentiment > 0.25:
            consensus = "Mostly Positive"
        elif avg_sentiment < -0.25:
            consensus = "Mostly Negative"
        else:
            consensus = "Mixed / Neutral"

        return {
            "average_sentiment": avg_sentiment,
            "consensus": consensus
        }

    def visualize(self, df):
        """Create a bar chart of persona sentiment."""
        fig = px.bar(
            df,
            x="persona",
            y="sentiment",
            title="Persona Sentiment Scores",
            text="sentiment",
        )
        fig.update_layout(yaxis_title="Sentiment Polarity (-1 to 1)")
        return fig

    def build_report(self, df, consensus_info):
        """Generate a human-readable summary."""
        summary = f"""
### Persona Feedback Summary
Average Sentiment: **{consensus_info['average_sentiment']:.2f}**
Consensus Level: **{consensus_info['consensus']}**

#### Individual Persona Notes:
"""

        for _, row in df.iterrows():
            summary += f"- **{row['persona']}** → Sentiment: {row['sentiment']:.2f}\n  - \"{row['feedback']}\"\n\n"

        return summary


# Main entry point used by Streamlit
def analyze_feedback():
    analyzer = FeedbackAnalyzer()
    df = analyzer.aggregate_feedback()
    consensus_info = analyzer.compute_consensus(df)
    fig = analyzer.visualize(df)
    report = analyzer.build_report(df, consensus_info)

    return df, consensus_info, fig, report
