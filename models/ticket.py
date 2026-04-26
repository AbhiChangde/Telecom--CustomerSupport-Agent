from typing import Literal, Optional
from pydantic import BaseModel


class Ticket(BaseModel):
    ticket_id: str
    issue_summary: str
    status: Literal["open", "closed_resolved", "closed_unresolved", "auto_closed"]
    created_date: str
    closed_date: Optional[str] = None
    issue_category: str
