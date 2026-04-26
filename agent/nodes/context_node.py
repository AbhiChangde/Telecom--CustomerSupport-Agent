import asyncio
from agent.state import SupportAgentState
from services.crm import get_customer_profile, get_tickets
from services.vector_db import recall_similar_complaints
from services.outage import check_outage


async def context_node(state: SupportAgentState) -> SupportAgentState:
    customer_id = state["customer_id"]
    user_input = state["user_input"]

    # Three calls fire simultaneously — total latency = slowest single call
    profile, tickets, vector_memory = await asyncio.gather(
        get_customer_profile(customer_id),
        get_tickets(customer_id),
        recall_similar_complaints(customer_id, user_input),
    )

    outage_status = await check_outage(profile.get("location", ""))

    prior_ticket_count = sum(
        1 for t in tickets
        if t["status"] in ("closed_unresolved", "auto_closed")
    )

    return {
        **state,
        "customer_profile": profile,
        "tickets": tickets,
        "prior_ticket_count": prior_ticket_count,
        "vector_memory": vector_memory,
        "outage_status": outage_status,
    }
