import pytest
from agent.conditions import route_resolution


def _make_state(action: str, frustration: str, prior_count: int, churn_risk: bool, user_input: str = ""):
    return {
        "decision": {
            "recommended_action": action,
            "frustration_level": frustration,
            "churn_risk": churn_risk,
        },
        "prior_ticket_count": prior_count,
        "user_input": user_input,
    }


def test_raise_ticket_route():
    state = _make_state("raise_ticket", "low", 0, False)
    assert route_resolution(state) == "raise_ticket"


def test_update_existing_route():
    state = _make_state("update_existing", "medium", 0, False)
    assert route_resolution(state) == "ticket_exists"


def test_escalate_human_route():
    state = _make_state("escalate_human", "high", 0, False)
    assert route_resolution(state) == "escalate"


def test_frustration_override_critical():
    """critical frustration always escalates regardless of recommended_action."""
    state = _make_state("raise_ticket", "critical", 0, False)
    assert route_resolution(state) == "escalate"


def test_frustration_override_high_with_prior():
    """high frustration + 2+ prior tickets escalates."""
    state = _make_state("raise_ticket", "high", 2, False)
    assert route_resolution(state) == "escalate"


def test_churn_risk_escalates():
    state = _make_state("raise_ticket", "low", 0, True)
    assert route_resolution(state) == "escalate"


def test_churn_word_in_input_escalates():
    state = _make_state("raise_ticket", "medium", 0, False, "I will switch to bsnl")
    assert route_resolution(state) == "escalate"


def test_high_frustration_single_prior_does_not_escalate():
    """high frustration with only 1 prior ticket does NOT auto-escalate (needs 2+)."""
    state = _make_state("raise_ticket", "high", 1, False)
    assert route_resolution(state) == "raise_ticket"
