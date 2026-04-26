import asyncio
import pytest
from agent.graph import support_graph
from agent.state import SupportAgentState


def _base_state(customer_id: str, text: str, language_code: str = "en") -> SupportAgentState:
    return SupportAgentState(
        customer_id=customer_id,
        language_code=language_code,
        language_name="English",
        user_input=text,
        input_mode="text",
        audio_duration_seconds=None,
        customer_profile=None,
        tickets=[],
        prior_ticket_count=0,
        vector_memory=[],
        outage_status="none",
        decision=None,
        is_frustrated=False,
        response_text="",
        new_ticket_id=None,
        handoff_brief=None,
        resolution_branch=None,
    )


@pytest.mark.asyncio
async def test_graph_returns_response_text():
    state = _base_state("CUST002", "My SIM is not working")
    result = await support_graph.ainvoke(state)
    assert result["response_text"], "response_text should not be empty"
    assert result["resolution_branch"] in ("raise_ticket", "ticket_exists", "escalate")


@pytest.mark.asyncio
async def test_graph_ticket_exists_for_cust003():
    """CUST003 has an open ticket — should route to ticket_exists."""
    state = _base_state("CUST003", "My bill is higher than my plan amount")
    result = await support_graph.ainvoke(state)
    # CUST003 has an open billing ticket; Gemini should recommend update_existing
    assert result["resolution_branch"] in ("ticket_exists", "escalate")


@pytest.mark.asyncio
async def test_graph_churn_word_triggers_escalation():
    state = _base_state("CUST002", "I am going to port to Jio if this is not fixed today")
    result = await support_graph.ainvoke(state)
    assert result["resolution_branch"] == "escalate"
    assert result["handoff_brief"] is not None


@pytest.mark.asyncio
async def test_graph_high_arpu_customer():
    state = _base_state("CUST001", "5G is not working for 3 days, this is pathetic")
    result = await support_graph.ainvoke(state)
    assert result["decision"]["priority_score"] >= 7, "High-ARPU customer should score high"
