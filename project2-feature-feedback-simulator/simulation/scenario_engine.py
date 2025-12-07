from tinytroupe.agent.tiny_person import TinyPerson
from tinytroupe.environment import TinyWorld

class ScenarioEngine:
    def __init__(self, persona: TinyPerson):
        self.persona = persona
        # Create a world
        self.world = TinyWorld()
        # Only add persona if not already registered to avoid duplicate name errors
        if persona.name not in TinyPerson.all_agents:
            self.world.add_agent(persona)

    def run_scenario(self, feature_description, scenario_type):
        # Build the prompt using direct attributes
        prompt = self._build_prompt(feature_description, scenario_type)
        return self.persona.listen_and_act(prompt)

    def _build_prompt(self, feature_description, scenario_type):
        # Scenario-specific instructions
        if scenario_type == "Onboarding":
            scenario_details = """
            You are a new user seeing this feature for the first time.
            Describe your initial reaction, confusion points, and whether the
            instructions are clear. Explain how difficult it is to learn.
            """
        elif scenario_type == "Feature Discovery":
            scenario_details = """
            Assume you discover this feature randomly while using the app.
            Describe whether you would understand it, ignore it, or use it.
            Explain your mental model, expectations, and interest level.
            """
        elif scenario_type == "Long-Term Usage":
            scenario_details = """
            Assume you have been using the app for 2 weeks and this new feature
            appears in an update. Describe whether it fits into your habits,
            whether it feels helpful or annoying, and whether you'd adopt it.
            """
        elif scenario_type == "Abandonment Risk":
            scenario_details = """
            Assume this feature becomes harder or requires extra steps.
            Describe whether you would stop using the feature or abandon the app.
            Identify points that cause frustration or confusion.
            """
        else:
            scenario_details = "Provide a realistic persona-specific reaction."

        # Build prompt using direct persona attributes
        prompt = f"""
        You are acting as the persona described below:

        Persona Details:
        - Name: {self.persona.name}
        - Age: {getattr(self.persona, 'age', 'N/A')}
        - Background: {getattr(self.persona, 'background', 'N/A')}
        - Technical proficiency: {getattr(self.persona, 'tech_level', 'N/A')}
        - Preferences: {getattr(self.persona, 'preferences', 'N/A')}
        - Behavioral tendencies: {getattr(self.persona, 'behavior', 'N/A')}

        Feature: {feature_description}

        Scenario: {scenario_type}
        {scenario_details}

        Respond with:
        - Persona reaction
        - Usability issues
        - Emotional response
        - Behavior changes
        - Likelihood of continued usage
        - Suggestions for improvement
        - Reasoning + confidence level
        """

        return prompt.strip()

