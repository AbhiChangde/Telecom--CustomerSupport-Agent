from agent.state import SupportAgentState

CHURN_WORDS = [
    "port", "jio", "vi", "bsnl", "vodafone", "airtel chhod",
    "consumer forum", "trai", "legal action", "nch", "done with",
    "switching", "never using", "last time", "bye airtel",
]


def is_critically_frustrated(state: SupportAgentState) -> bool:
    decision = state.get("decision") or {}
    frustration = decision.get("frustration_level", "low")
    prior_count = state.get("prior_ticket_count", 0)
    user_input = state.get("user_input", "").lower()

    return any([
        frustration == "critical",
        frustration == "high" and prior_count >= 2,
        any(word in user_input for word in CHURN_WORDS),
        decision.get("churn_risk", False),
    ])


def route_resolution(state: SupportAgentState) -> str:
    if is_critically_frustrated(state):
        return "escalate"

    action = (state.get("decision") or {}).get("recommended_action", "raise_ticket")

    if action == "escalate_human":
        return "escalate"
    elif action == "update_existing":
        return "ticket_exists"
    else:
        return "raise_ticket"


def compute_priority_tier(score: int) -> str:
    if score >= 8:
        return "P1"
    if score >= 5:
        return "P2"
    return "P3"


def compute_priority_score(
    arpu_tier: str,
    tenure_years: float,
    prior_unresolved: int,
    frustration_level: str,
) -> int:
    """Deterministic fallback scorer if Gemini returns invalid score."""
    arpu_scores = {
        "black": 10,
        "postpaid_high": 8,
        "postpaid_std": 6,
        "prepaid_annual": 4,
        "prepaid_monthly": 2,
    }
    arpu_score = arpu_scores.get(arpu_tier, 4)

    tenure_score = 10 if tenure_years >= 5 else 6 if tenure_years >= 2 else 2

    prior_score = (
        10 if prior_unresolved >= 3
        else 7 if prior_unresolved == 2
        else 4 if prior_unresolved == 1
        else 1
    )

    frustration_scores = {"critical": 10, "high": 7, "medium": 4, "low": 1}
    frustration_score = frustration_scores.get(frustration_level, 1)

    weighted = (
        arpu_score * 0.35
        + tenure_score * 0.20
        + prior_score * 0.30
        + frustration_score * 0.15
    )
    return min(10, max(1, round(weighted)))
