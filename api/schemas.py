from typing import Optional, Literal
from pydantic import BaseModel


class StartSessionResponse(BaseModel):
    session_id: str
    customer_id: str
    language_code: str
    greeting_url: str


class AgentResponse(BaseModel):
    response: str
    resolution_branch: Optional[Literal["raise_ticket", "ticket_exists", "escalate"]]
    priority_score: Optional[int]
    priority_tier: Optional[str]
    intent: Optional[str] = None
    transcript: Optional[str] = None
    ticket_id: Optional[str] = None
    handoff_brief: Optional[dict] = None
