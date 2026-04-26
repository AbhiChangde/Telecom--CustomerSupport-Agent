import json
from agent.state import SupportAgentState
from agent.conditions import compute_priority_score
from services.gemini import call_gemini

DECISION_PROMPT = """
You are the decision engine for a telecom customer support agent.
Analyse the customer's complaint and the context provided.
Return ONLY valid JSON — no markdown, no explanation.

Customer said (in {language}):
"{user_input}"

Customer context:
- ARPU tier: {arpu_tier} (monthly: ₹{arpu_monthly})
- Tenure: {tenure_years} years
- Plan: {plan_name}
- Prior tickets on any issue: {total_tickets}
- Prior unresolved tickets on same type of issue: {prior_ticket_count}
- Open tickets: {open_tickets}
- Outage in area: {outage_status}
- Similar past complaints found: {vector_memory_summary}

CRITICAL RULE FOR "intent" FIELD:
- ALWAYS write the intent in ENGLISH, regardless of what language the customer used.
- It must be a short issue category of 3-6 words only.
- Use telecom category language. Good examples:
    "5G connectivity not working"
    "Data balance depleting fast"
    "Postpaid bill overcharge"
    "SIM not activated"
    "Call dropping frequently"
    "WiFi calling not working"
    "Network outage in area"
    "Recharge not reflected"
- NEVER copy or translate the customer's words. Classify the issue into a category.

Return this exact JSON structure:
{{
  "intent": "<3-6 word English issue category>",
  "frustration_level": "low|medium|high|critical",
  "frustration_signals": ["list of specific signals detected"],
  "ticket_status": "none|open|closed_unresolved",
  "recommended_action": "raise_ticket|update_existing|escalate_human",
  "priority_score": <integer 1-10>,
  "priority_reasoning": "brief explanation of score",
  "churn_risk": <true|false>
}}

Priority scoring guide:
- ARPU tier weight (35%): black=10, postpaid_high=8, postpaid_std=6, prepaid_annual=4, prepaid_monthly=2
- Tenure weight (20%): >5yr=10, 2-5yr=6, <1yr=2
- Prior unresolved complaints weight (30%): 3+=10, 2=7, 1=4, 0=1
- Frustration weight (15%): critical=10, high=7, medium=4, low=1
Compute weighted average and scale to 1-10.
"""

# ASCII character range — used to detect if intent leaked non-English text
def _looks_english(text: str) -> bool:
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / max(len(text), 1) < 0.2


async def decision_node(state: SupportAgentState) -> SupportAgentState:
    profile = state["customer_profile"]
    tickets = state["tickets"]
    open_tickets = [t for t in tickets if t["status"] == "open"]
    vector_summary = (
        f"{len(state['vector_memory'])} similar past complaints found"
        if state["vector_memory"]
        else "none"
    )

    prompt = DECISION_PROMPT.format(
        language=state["language_name"],
        user_input=state["user_input"],
        arpu_tier=profile["arpu_tier"],
        arpu_monthly=profile["arpu_monthly"],
        tenure_years=profile["tenure_years"],
        plan_name=profile["plan_name"],
        total_tickets=len(tickets),
        prior_ticket_count=state["prior_ticket_count"],
        open_tickets=len(open_tickets),
        outage_status=state["outage_status"],
        vector_memory_summary=vector_summary,
    )

    raw = await call_gemini(prompt)

    # Strip markdown fences if Gemini wraps in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        decision = json.loads(raw)
    except json.JSONDecodeError:
        frustration = "medium"
        decision = {
            "intent": "Network or service issue",
            "frustration_level": frustration,
            "frustration_signals": [],
            "ticket_status": "open" if open_tickets else "none",
            "recommended_action": "raise_ticket",
            "priority_score": compute_priority_score(
                profile["arpu_tier"],
                profile["tenure_years"],
                state["prior_ticket_count"],
                frustration,
            ),
            "priority_reasoning": "Fallback score — Gemini returned unparseable response",
            "churn_risk": False,
        }

    # Safety: if intent contains non-English text, replace with a generic category
    intent = decision.get("intent", "")
    if not _looks_english(intent) or len(intent) > 60:
        decision["intent"] = "Network or service issue"

    # Clamp priority score to 1–10
    decision["priority_score"] = max(1, min(10, int(decision.get("priority_score", 5))))

    is_frustrated = decision.get("frustration_level") in ("high", "critical")

    return {**state, "decision": decision, "is_frustrated": is_frustrated}
