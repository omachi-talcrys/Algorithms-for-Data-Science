from tinytroupe.agent.tiny_person import TinyPerson
from tinytroupe.environment import TinyWorld

class BatchEngine:
    def __init__(self, personas):
        """
        personas: list of TinyPerson objects
        """
        self.personas = personas
        self.world = TinyWorld()
        # Add agents only if their name isn't already in use
        for persona in personas:
            if persona.name not in TinyPerson.all_agents:
                self.world.add_agent(persona)

    def run_batch(self, feature_description, scenario_prompt=""):
        """
        Run feature simulation for all personas.
        Returns a dict with persona names as keys and feedback as values.
        """
        results = {}

        for persona in self.personas:
            prompt = self._build_prompt(persona, feature_description, scenario_prompt)
            feedback = persona.listen_and_act(prompt)
            results[persona.name] = feedback

        return results

    def _build_prompt(self, persona, feature_description, scenario_prompt):
        """
        Build a persona-specific prompt including optional scenario.
        Uses direct attributes to avoid memory issues.
        """
        prompt = f"""
        You are acting as the persona described below.

        Persona Details:
        - Name: {persona.name}
        - Age: {getattr(persona, 'age', 'N/A')}
        - Background: {getattr(persona, 'background', 'N/A')}
        - Technical proficiency: {getattr(persona, 'tech_level', 'N/A')}
        - Preferences: {getattr(persona, 'preferences', 'N/A')}
        - Behavioral tendencies: {getattr(persona, 'behavior', 'N/A')}

        Feature Description:
        {feature_description}

        Scenario:
        {scenario_prompt if scenario_prompt else 'General usage scenario'}

        Please respond with:
        - Persona reaction
        - Usability concerns
        - Learning curve or difficulty
        - Likelihood of continued usage
        - Emotional response
        - Suggestions for improvement
        - Your reasoning and confidence level
        """
        return prompt.strip()



