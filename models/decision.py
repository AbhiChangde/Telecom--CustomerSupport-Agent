from typing import Literal
from pydantic import BaseModel, field_validator


class GeminiDecision(BaseModel):
    intent: str
    frustration_level: Literal["low", "medium", "high", "critical"]
    frustration_signals: list[str]
    ticket_status: Literal["none", "open", "closed_unresolved"]
    recommended_action: Literal["raise_ticket", "update_existing", "escalate_human"]
    priority_score: int
    priority_reasoning: str
    churn_risk: bool

    @field_validator("priority_score")
    @classmethod
    def clamp_priority(cls, v: int) -> int:
        return max(1, min(10, v))
