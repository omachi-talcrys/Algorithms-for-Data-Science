import os
import time

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "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"

import tinytroupe_config
import streamlit as st
from tinytroupe.agent.tiny_person import TinyPerson
TinyPerson.all_agents.clear()


# Your project modules
from personas.predefined import load_predefined_personas
from personas.custom_creator import create_custom_persona
from analysis.aggregator import FeedbackAnalyzer
from analysis.visualization import FeedbackVisualizer
from simulation.scenario_engine import ScenarioEngine
from simulation.batch_engine import BatchEngine
from reports.exporter import ReportExporter



# Helper: run function with timeout
def run_with_timeout(func, timeout=45):
    start = time.time()
    result = None
    exception = None

    try:
        result = func()
    except Exception as e:
        exception = e

    elapsed = time.time() - start
    if elapsed > timeout:
        raise TimeoutError("Agent call exceeded time limit")
    if exception:
        raise exception
    return result

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("AI Persona Feature Feedback Simulator")
st.write("Generate persona-based feedback for any feature using AI simulation.")

# Load personas
personas = load_predefined_personas()
persona_choice = st.selectbox("Choose a Persona", ["Select...", "Predefined", "Custom Persona"])
selected_persona = None

# Persona selection
if persona_choice == "Predefined":
    persona_name = st.selectbox("Select Predefined Persona", list(personas.keys()))
    selected_persona = personas.get(persona_name)

elif persona_choice == "Custom Persona":
    st.subheader("Create Custom Persona")
    custom_name = st.text_input("Name")
    custom_age = st.text_input("Age")
    custom_background = st.text_area("Background")
    custom_tech_level = st.text_input("Technical Proficiency Level")
    custom_preferences = st.text_area("Preferences")
    custom_behavior = st.text_area("Behavioral Tendencies")

    if st.button("Create Persona"):
        selected_persona = create_custom_persona(
            custom_name, custom_age, custom_background,
            custom_tech_level, custom_preferences, custom_behavior
        )
        st.success(f"Custom persona '{custom_name}' created!")

# Feature input
st.subheader("Feature Description")
feature_description = st.text_area("Describe the feature (interaction, visuals, flow, purpose)...")

# Initialize analytics
aggregator = FeedbackAnalyzer()
visualizer = FeedbackVisualizer()

# ------------------------------
# Single Persona Feedback
# ------------------------------
if st.button("Simulate Feedback"):
    if not selected_persona or not feature_description.strip():
        st.error("❗ Please select a persona and enter a feature description.")
    else:
        # Build prompt
        prompt = f"""
You are {selected_persona.name}, a persona with the following profile:

Persona Profile:
- Age: {getattr(selected_persona, 'age', 'N/A')}
- Background: {getattr(selected_persona, 'background', 'N/A')}
- Technical proficiency: {getattr(selected_persona, 'tech_level', 'N/A')}
- Preferences: {getattr(selected_persona, 'preferences', 'N/A')}
- Behavioral tendencies: {getattr(selected_persona, 'behavior', 'N/A')}

New Feature Being Evaluated:
{feature_description}

Respond as this persona would with clearly separated sections:
- Initial reaction
- Usability concerns
- Likelihood of using the feature
- Likes / dislikes
- Suggestions for improvement
- Reasoning and confidence
"""

        # Show user message immediately
        st.chat_message("user").write(prompt)

        # Placeholder for AI response
        response_placeholder = st.empty()
        response_placeholder.text("🧠 Persona is thinking...")

        try:
            actions = run_with_timeout(
                lambda: selected_persona.listen_and_act(prompt, return_actions=True),
                timeout=45
            )

            # Extract only the persona's speech
            response = "\n".join([a.content for a in actions if a.role == "TALK"])

            if not response.strip():
                response_placeholder.text("⚠️ Persona returned no visible response.")
            else:
                response_placeholder.text("")  # clear placeholder
                st.chat_message("assistant").write(response)

                # Run analytics
                try:
                    df, consensus_info, fig, report = analyze_feedback({selected_persona.name: response})
                    st.plotly_chart(fig)
                    st.markdown(report)
                except Exception as e:
                    st.warning(f"Analysis failed but feedback was generated: {e}")

        except TimeoutError:
            response_placeholder.text("⏰ AI took too long to respond.")
        except Exception as e:
            response_placeholder.text(f"❌ AI call failed: {e}")

# ------------------------------
# Scenario Simulation
# ------------------------------
st.subheader("Scenario Simulation")
scenario_type = st.selectbox(
    "Choose a scenario",
    ["None", "Onboarding", "Feature Discovery", "Long-Term Usage", "Abandonment Risk"]
)

if st.button("Run Scenario Simulation"):
    if not selected_persona or not feature_description.strip() or scenario_type == "None":
        st.error("❗ Select persona, enter feature, and choose scenario.")
    else:
        engine = ScenarioEngine(selected_persona)
        scenario_feedback = engine.run_scenario(feature_description, scenario_type)
        aggregator.add_feedback(f"{selected_persona.name} ({scenario_type})", scenario_feedback)
        st.chat_message("assistant").write(scenario_feedback)
        scenario_trends = aggregator.compute_scenario_trends({
            selected_persona.name: {scenario_type: scenario_feedback}
        })
        st.bar_chart(scenario_trends)

# ------------------------------
# Batch Persona Testing
# ------------------------------
st.subheader("Batch Persona Testing")
selected_batch_personas = st.multiselect("Select multiple personas for batch testing", options=list(personas.keys()))
batch_scenario_type = st.selectbox(
    "Optional Scenario for Batch Testing",
    ["None", "Onboarding", "Feature Discovery", "Long-Term Usage", "Abandonment Risk"],
    index=0
)

if st.button("Run Batch Simulation"):
    if not feature_description.strip() or len(selected_batch_personas) < 1:
        st.error("❗ Enter feature description and select at least one persona.")
    else:
        batch_persona_objs = [personas[p] for p in selected_batch_personas]
        batch_engine = BatchEngine(batch_persona_objs)
        scenario_prompt = ""
        if batch_scenario_type != "None":
            scenario_engine = ScenarioEngine(batch_persona_objs[0])
            scenario_prompt = scenario_engine._build_prompt(
                batch_persona_objs[0], feature_description, batch_scenario_type
            )
        results = batch_engine.run_batch(feature_description, scenario_prompt)
        for pname, feedback in results.items():
            key_name = f"{pname} ({batch_scenario_type if batch_scenario_type != 'None' else 'General'})"
            aggregator.add_feedback(key_name, feedback)
            st.markdown(f"**{pname} Feedback:**")
            st.chat_message("assistant").write(feedback)

        visualizer.sentiment_bar_chart(aggregator.compute_sentiment_scores())
        visualizer.theme_list(aggregator.extract_themes())
        visualizer.combined_summary(aggregator.combined_summary())
        visualizer.radar_chart(aggregator.compute_persona_scores())
        visualizer.frequency_chart(aggregator.compute_word_frequency())

# ------------------------------
# Export Reports
# ------------------------------
st.subheader("Export Reports")
export_format = st.selectbox("Choose export format:", ["None", "CSV", "PDF", "Text"])

if st.button("Export"):
    if not feature_description.strip():
        st.error("❗ Cannot export without feedback or results.")
    else:
        export_data = {}
        if "results" in locals():
            export_data = results
        elif "scenario_feedback" in locals():
            export_data = {"Scenario Feedback": scenario_feedback}
        elif "response" in locals():
            export_data = {"Persona Feedback": response}

        if not export_data:
            st.error("❗ No data to export yet.")
        else:
            if export_format == "CSV":
                file_path = ReportExporter.export_csv(export_data, "feedback_report")
                st.success(f"CSV saved to: {file_path}")
            elif export_format == "PDF":
                combined_text = ""
                for k, v in export_data.items():
                    combined_text += f"--- {k} ---\n{v}\n\n"
                file_path = ReportExporter.export_pdf("Feature Feedback Report", combined_text, "feedback_report")
                st.success(f"PDF saved to: {file_path}")
            elif export_format == "Text":
                combined_text = ""
                for k, v in export_data.items():
                    combined_text += f"{k}:\n{v}\n\n"
                file_path = ReportExporter.save_txt(combined_text, "feedback_report")
                st.success(f"Text file saved to: {file_path}")
            else:
                st.error("❗ Choose a valid export format.")
