from typing import TypedDict, Optional, Literal


class CustomerProfile(TypedDict):
    customer_id: str
    name: str
    mobile: str
    arpu_monthly: float
    arpu_tier: Literal["black", "postpaid_high", "postpaid_std", "prepaid_annual", "prepaid_monthly"]
    tenure_years: float
    plan_type: Literal["postpaid", "prepaid"]
    plan_name: str
    location: str


class Ticket(TypedDict):
    ticket_id: str
    issue_summary: str
    status: Literal["open", "closed_resolved", "closed_unresolved", "auto_closed"]
    created_date: str
    closed_date: Optional[str]
    issue_category: str


class GeminiDecision(TypedDict):
    intent: str
    frustration_level: Literal["low", "medium", "high", "critical"]
    frustration_signals: list[str]
    ticket_status: Literal["none", "open", "closed_unresolved"]
    recommended_action: Literal["raise_ticket", "update_existing", "escalate_human"]
    priority_score: int
    priority_reasoning: str
    churn_risk: bool


class SupportAgentState(TypedDict):
    # State 1 — Language
    customer_id: str
    language_code: str
    language_name: str

    # State 3 — Input
    user_input: str
    input_mode: Literal["voice", "text"]
    audio_duration_seconds: Optional[float]

    # State 4 — Context
    customer_profile: Optional[CustomerProfile]
    tickets: list[Ticket]
    prior_ticket_count: int
    vector_memory: list[dict]
    outage_status: str

    # State 5 — Decision
    decision: Optional[GeminiDecision]
    is_frustrated: bool

    # State 6 — Resolution
    response_text: str
    new_ticket_id: Optional[str]
    handoff_brief: Optional[dict]
    resolution_branch: Optional[Literal["raise_ticket", "ticket_exists", "escalate"]]
