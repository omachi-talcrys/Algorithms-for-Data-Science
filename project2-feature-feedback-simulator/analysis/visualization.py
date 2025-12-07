import streamlit as st
import pandas as pd

class FeedbackVisualizer:

    def sentiment_bar_chart(self, sentiment_dict):
        st.subheader("Sentiment Scores by Persona")
        df = pd.DataFrame({
            "Persona": list(sentiment_dict.keys()),
            "Sentiment Score": list(sentiment_dict.values())
        })
        st.bar_chart(df.set_index("Persona"))

    def theme_list(self, themes):
        st.subheader("Extracted Themes Across Personas")
        for theme, words in themes.items():
            st.markdown(f"### {theme.title()}")
            st.write(", ".join(words))

    def combined_summary(self, summary):
        st.subheader("Overall Combined Feedback Summary")
        st.write(summary["combined_feedback"])

