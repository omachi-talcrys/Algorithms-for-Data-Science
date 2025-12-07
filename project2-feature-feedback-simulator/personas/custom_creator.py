from tinytroupe.agent.tiny_person import TinyPerson

def create_custom_persona(name, age, background, tech, preferences, behavior):
    # ✅ Clear duplicates safely
    if name in TinyPerson.all_agents:
        del TinyPerson.all_agents[name]

    persona = TinyPerson(name=name)

    # ✅ Store attributes directly on the object
    persona.age = age
    persona.background = background
    persona.tech_level = tech
    persona.preferences = preferences
    persona.behavior = behavior

    return persona


