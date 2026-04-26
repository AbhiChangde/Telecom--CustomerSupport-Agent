from langgraph.graph import StateGraph, END
from agent.state import SupportAgentState
from agent.nodes.language_node import language_node
from agent.nodes.greeting_node import greeting_node
from agent.nodes.input_node import input_node
from agent.nodes.context_node import context_node
from agent.nodes.decision_node import decision_node
from agent.nodes import resolution_node
from agent.conditions import route_resolution


def _wrap(fn):
    """Compatibility shim: newer LangGraph passes state as **kwargs; older passes as positional arg."""
    async def wrapped(*args, **kwargs):
        state = args[0] if args else kwargs
        return await fn(state)
    return wrapped


def build_graph() -> StateGraph:
    graph = StateGraph(SupportAgentState)

    graph.add_node("language",     _wrap(language_node))
    graph.add_node("greeting",     _wrap(greeting_node))
    graph.add_node("input",        _wrap(input_node))
    graph.add_node("context",      _wrap(context_node))
    graph.add_node("decision",     _wrap(decision_node))
    graph.add_node("raise_ticket", _wrap(resolution_node.raise_ticket))
    graph.add_node("ticket_exists",_wrap(resolution_node.ticket_exists))
    graph.add_node("escalate",     _wrap(resolution_node.escalate))

    graph.set_entry_point("language")
    graph.add_edge("language", "greeting")
    graph.add_edge("greeting", "input")
    graph.add_edge("input", "context")
    graph.add_edge("context", "decision")

    graph.add_conditional_edges(
        "decision",
        route_resolution,
        {
            "raise_ticket": "raise_ticket",
            "ticket_exists": "ticket_exists",
            "escalate": "escalate",
        },
    )

    graph.add_edge("raise_ticket", END)
    graph.add_edge("ticket_exists", END)
    graph.add_edge("escalate", END)

    return graph.compile()


support_graph = build_graph()
