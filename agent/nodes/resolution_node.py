from agent.state import SupportAgentState
from agent.conditions import compute_priority_tier
from services.crm import raise_ticket as crm_raise_ticket
from services.sms import send_sms
from services.vector_db import store_conversation


def _generate_first_move(decision: dict, tickets: list) -> str:
    if decision.get("churn_risk"):
        return (
            "CHURN RISK — Do NOT use a script. Acknowledge complaint history by date. "
            "Retention offer pre-authorised. Last chance contact."
        )
    prior_count = decision.get("prior_ticket_count", 0)
    # Use tickets list length as proxy when prior_ticket_count not in decision
    unresolved = sum(1 for t in tickets if t["status"] in ("closed_unresolved", "auto_closed"))
    if unresolved >= 2:
        return (
            f"Customer has {unresolved} unresolved tickets on this issue. "
            "Do NOT ask them to explain again. Acknowledge history immediately."
        )
    return (
        f"Issue: {decision['intent']}. "
        f"Tone is {decision['frustration_level']}. "
        "Proceed with standard resolution but acknowledge wait time."
    )


async def raise_ticket_branch(state: SupportAgentState) -> SupportAgentState:
    profile = state["customer_profile"]
    decision = state["decision"]

    ticket_id = await crm_raise_ticket(
        customer_id=state["customer_id"],
        issue=decision["intent"],
        priority=decision["priority_score"],
    )
    await send_sms(
        profile["mobile"],
        f"Your complaint has been registered. Ticket: {ticket_id}. "
        "Our team will contact you within 24 hours.",
    )
    await store_conversation(state["customer_id"], state["user_input"], decision["intent"])

    response = (
        f"Your complaint has been registered. Ticket ID: {ticket_id}. "
        "You'll receive an SMS confirmation shortly. Our team will reach out within 24 hours."
    )

    return {**state, "response_text": response, "new_ticket_id": ticket_id, "resolution_branch": "raise_ticket"}


async def ticket_exists_branch(state: SupportAgentState) -> SupportAgentState:
    open_ticket = next((t for t in state["tickets"] if t["status"] == "open"), None)

    if open_ticket:
        response = (
            f"I can see you already have an open ticket ({open_ticket['ticket_id']}) "
            f"for this issue raised on {open_ticket['created_date']}. "
            "Our team is actively working on it. You'll be contacted within the committed timeline."
        )
    else:
        # Fallback — no open ticket found despite routing here; raise a new one
        return await raise_ticket_branch(state)

    return {**state, "response_text": response, "resolution_branch": "ticket_exists"}


async def escalate_branch(state: SupportAgentState) -> SupportAgentState:
    profile = state["customer_profile"]
    decision = state["decision"]
    tier = compute_priority_tier(decision["priority_score"])

    brief = {
        "customer_name": profile["name"],
        "mobile": profile["mobile"],
        "arpu_tier": profile["arpu_tier"],
        "plan": profile["plan_name"],
        "tenure_years": profile["tenure_years"],
        "priority_tier": tier,
        "priority_score": decision["priority_score"],
        "parsed_issue": decision["intent"],
        "verbatim_complaint": state["user_input"],
        "frustration_level": decision["frustration_level"],
        "frustration_signals": decision["frustration_signals"],
        "churn_risk": decision["churn_risk"],
        "ticket_history": state["tickets"],
        "prior_unresolved": state["prior_ticket_count"],
        "outage_status": state["outage_status"],
        "first_move": _generate_first_move(decision, state["tickets"]),
    }

    if decision.get("churn_risk"):
        response = (
            "I completely understand your frustration. "
            "I'm connecting you with a senior specialist right now who has your full history "
            "and the authority to resolve this immediately. "
            "You will not need to explain anything again."
        )
    else:
        response = (
            "I'm escalating this to our support team who will call you back shortly. "
            "They already have your full details and complaint history."
        )

    return {**state, "response_text": response, "handoff_brief": brief, "resolution_branch": "escalate"}


# Export branch map for graph wiring
raise_ticket = raise_ticket_branch
ticket_exists = ticket_exists_branch
escalate = escalate_branch
