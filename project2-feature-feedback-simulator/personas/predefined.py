from .custom_creator import create_custom_persona

def make_persona(name, age, background, tech_level, preferences, behavior):
    return create_custom_persona(
        name=name,
        age=age,
        background=background,
        tech=tech_level,
        preferences=preferences,
        behavior=behavior
    )

def load_predefined_personas():
    return {
        "Tech-Savvy Young Adult": make_persona(
            name="Jordan",
            age="26",
            background="Heavy smartphone user, early adopter.",
            tech_level="Very high",
            preferences="Minimal UI, fast navigation.",
            behavior="Gets annoyed by slow/confusing flows."
        ),
        "Casual Everyday User": make_persona(
            name="Alicia",
            age="40",
            background="Uses phone for messaging, browsing, shopping.",
            tech_level="Moderate",
            preferences="Clear instructions, simple layout.",
            behavior="Avoids complex settings."
        ),
        "Elderly User": make_persona(
            name="Mr. Thompson",
            age="72",
            background="Low smartphone familiarity.",
            tech_level="Low",
            preferences="Large buttons, high contrast.",
            behavior="Slow learner."
        ),
        "Accessibility User": make_persona(
            name="Rina",
            age="33",
            background="Blind/low-vision, depends on screen readers.",
            tech_level="High",
            preferences="VoiceOver, clear labels.",
            behavior="Abandons inaccessible features."
        )
    }
