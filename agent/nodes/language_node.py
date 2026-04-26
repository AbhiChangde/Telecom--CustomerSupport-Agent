from agent.state import SupportAgentState

LANGUAGE_NAMES = {
    "hi": "Hindi",
    "en": "English",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "bn": "Bengali",
}


async def language_node(state: SupportAgentState) -> SupportAgentState:
    language_code = state.get("language_code", "en")
    language_name = LANGUAGE_NAMES.get(language_code, "English")
    return {**state, "language_code": language_code, "language_name": language_name}
