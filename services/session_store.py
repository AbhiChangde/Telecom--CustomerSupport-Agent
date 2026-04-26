from agent.conversation import ConversationAgent

_sessions: dict[str, ConversationAgent] = {}


def create(session_id: str, customer_id: str, language_code: str, language_name: str) -> ConversationAgent:
    agent = ConversationAgent(session_id, customer_id, language_code, language_name)
    _sessions[session_id] = agent
    return agent


def get(session_id: str) -> ConversationAgent | None:
    return _sessions.get(session_id)


def delete(session_id: str) -> None:
    _sessions.pop(session_id, None)
