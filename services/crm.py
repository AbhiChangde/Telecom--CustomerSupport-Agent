import json
import os
import uuid
from pathlib import Path
from agent.state import CustomerProfile, Ticket

MOCK_DIR = Path(__file__).parent.parent / "data" / "mock"


def _load_customers() -> dict:
    with open(MOCK_DIR / "customers.json") as f:
        return json.load(f)


def _load_tickets() -> dict:
    with open(MOCK_DIR / "tickets.json") as f:
        return json.load(f)


async def get_customer_profile(customer_id: str) -> CustomerProfile:
    customers = _load_customers()
    if customer_id not in customers:
        raise ValueError(f"Customer {customer_id} not found")
    return customers[customer_id]


async def get_tickets(customer_id: str) -> list[Ticket]:
    tickets = _load_tickets()
    return tickets.get(customer_id, [])


async def raise_ticket(customer_id: str, issue: str, priority: int) -> str:
    if os.getenv("USE_MOCK_CRM", "true").lower() == "true":
        ticket_id = f"TKT-{customer_id[-4:]}-{uuid.uuid4().hex[:6].upper()}"
        print(f"[MOCK CRM] Raised ticket {ticket_id} for {customer_id}: {issue} (priority {priority})")
        return ticket_id

    raise NotImplementedError("Real CRM integration not implemented")


async def get_ticket_status(ticket_id: str) -> str:
    if os.getenv("USE_MOCK_CRM", "true").lower() == "true":
        return "open"
    raise NotImplementedError("Real CRM integration not implemented")
