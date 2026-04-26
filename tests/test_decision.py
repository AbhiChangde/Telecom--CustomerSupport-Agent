import pytest
from agent.conditions import compute_priority_score, compute_priority_tier, is_critically_frustrated


def test_priority_score_black_long_tenure():
    score = compute_priority_score("black", 6.0, 0, "low")
    # black(10)*0.35 + tenure≥5(10)*0.20 + prior=0(1)*0.30 + low(1)*0.15
    # = 3.5 + 2.0 + 0.3 + 0.15 = 5.95 → round = 6
    assert score == 6


def test_priority_score_high_frustration_unresolved():
    score = compute_priority_score("postpaid_high", 3.0, 2, "high")
    # 8*0.35 + 6*0.20 + 7*0.30 + 7*0.15
    # = 2.8 + 1.2 + 2.1 + 1.05 = 7.15 → round = 7
    assert score == 7


def test_priority_tier():
    assert compute_priority_tier(8) == "P1"
    assert compute_priority_tier(5) == "P2"
    assert compute_priority_tier(4) == "P3"
    assert compute_priority_tier(10) == "P1"


def test_score_clamped_to_range():
    score = compute_priority_score("black", 10.0, 3, "critical")
    assert 1 <= score <= 10


def test_is_critically_frustrated_churn_word():
    state = {
        "decision": {"frustration_level": "low", "churn_risk": False},
        "prior_ticket_count": 0,
        "user_input": "I will port to jio tomorrow",
    }
    assert is_critically_frustrated(state) is True


def test_is_critically_frustrated_critical_level():
    state = {
        "decision": {"frustration_level": "critical", "churn_risk": False},
        "prior_ticket_count": 0,
        "user_input": "This is very bad",
    }
    assert is_critically_frustrated(state) is True


def test_is_not_frustrated_low():
    state = {
        "decision": {"frustration_level": "low", "churn_risk": False},
        "prior_ticket_count": 0,
        "user_input": "My internet is slow",
    }
    assert is_critically_frustrated(state) is False
