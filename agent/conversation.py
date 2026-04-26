import json
import re
import os
import asyncio
import google.generativeai as genai

from services.crm import get_customer_profile, get_tickets, raise_ticket as crm_raise_ticket
from services.vector_db import store_conversation
from services.sms import send_sms
from agent.conditions import compute_priority_tier

# ── Frustration keyword sets ──────────────────────────────────────────────────
_HIGH_FRUSTRATION = {
    "useless", "worst", "terrible", "pathetic", "horrible", "awful",
    "fed up", "ridiculous", "fraud", "scam", "incompetent", "disgusting",
    "not working", "still not", "again", "always", "never works",
    "बेकार", "घटिया", "बेशर्म", "बकवास",
}
_CRITICAL_FRUSTRATION = {
    "legal action", "consumer forum", "trai", "police complaint",
    "done with airtel", "never using airtel", "switching",
    "port", "jio", "vi ", "bsnl", "vodafone", "cancel",
    "waste of money", "demanding refund",
}

SYSTEM_PROMPT_TEMPLATE = """You are Aria, a warm, empathetic and professional Airtel customer support voice agent. You are female.
Use she/her pronouns if you refer to yourself in third person.
This is a spoken conversation — keep EVERY response to 1-2 short sentences. Never use bullet points, markdown, or lists.
Always respond in {language_name}.

Customer profile (already loaded — NEVER ask for these details again):
- Name: {first_name}
- Plan: {plan_name} (₹{arpu_monthly}/month, {arpu_tier} tier)
- Tenure: {tenure_years} years with Airtel
- Prior unresolved tickets: {prior_unresolved}
- Open tickets right now: {open_tickets}

CUSTOMER TIER HANDLING — strictly follow these escalation rules based on {arpu_tier}:

black / postpaid_high (PREMIUM):
  These customers receive priority human support.
  If frustration reaches medium, high, or critical → escalate to a human specialist immediately.
  Do NOT attempt ticket self-service for a frustrated premium customer.

postpaid_std / prepaid_annual / prepaid_monthly (STANDARD):
  These customers are served through ticket-based resolution first.
  Raise a ticket and give STRONG, WARM ASSURANCE that the issue will be prioritised and resolved.
  Escalate to a human ONLY if frustration is high or critical (clearly agitated, aggressive, or repeated distress).
  Medium frustration → calm them and raise a ticket with assurance. Do NOT escalate.
  Low frustration → raise a ticket directly.

FRUSTRATION RESPONSE RULES — adapt based on the [Frustration] tag in each message:
- [Frustration: low] → proceed with normal resolution flow
- [Frustration: medium] → acknowledge warmly before resolving; for standard tier, raise ticket with assurance
- [Frustration: high] → spend one FULL TURN calming the customer BEFORE resolving; for standard tier, calm + raise ticket + strong assurance (do not escalate); for premium tier, escalate
- [Frustration: critical] → immediate empathy; escalate for ALL tiers

CONVERSATION FLOW:

1. GREET (on START only): Greet by first name. Acknowledge loyalty if tenure > 3 years. Ask what you can help with.

2. UNDERSTAND: Ask ONE clarifying question if needed (how long? which service?). Never ask what the customer already answered.

3. EMPATHISE: Validate frustration before any resolution.
   If prior unresolved tickets ({prior_unresolved}): "I can see this has been ongoing — I'm truly sorry we haven't sorted it sooner."

4. CALM (if frustration is high/critical): One warm reassuring sentence before acting.
   "I personally want to make sure this gets the attention it deserves."
   Do NOT skip this if frustration is high.

5. RESOLVE — decide action AND output RESOLVE IN THE SAME MESSAGE:
   - Existing open ticket → tell the customer the ticket ID AND output RESOLVE (update_existing)
   - New issue → tell the customer you are raising a ticket AND output RESOLVE (raise_ticket)
   - Critical frustration / churn risk / premium + medium+ frustration / 2+ unresolved → tell customer you are escalating AND output RESOLVE (escalate_human)

   *** CRITICAL RULE: The RESOLVE block MUST appear in the SAME message where you announce the action.
   If you say "I'm raising a ticket", RESOLVE must be in THAT message.
   NEVER say "I'll raise a ticket" in one message and put RESOLVE in a later message. ***

6. FOLLOW UP (after raise_ticket or update_existing): The system will confirm the action. Then ask:
   "Is there anything else I can help you with today?"
   - Satisfied → output RESOLVE (close)
   - New issue → go back to step 2
   - Still frustrated → calm, then escalate if demanded

MANDATORY: Every response MUST end with a question OR a clear action statement. Never leave the customer with nothing to reply to.

RESOLVE BLOCK (append at end of message when taking action or closing):
RESOLVE:{{"intent":"<3-6 word English issue>","frustration_level":"low|medium|high|critical","recommended_action":"raise_ticket|update_existing|escalate_human|close","priority_score":<1-10>,"churn_risk":<true|false>,"frustration_signals":["signal1","signal2"]}}

RESOLVE rules:
- intent: English only, 3-6 words (e.g. "5G connectivity failure", "Postpaid bill dispute")
- recommended_action: raise_ticket | update_existing | escalate_human | close
- priority_score: 1-10 (higher for black/postpaid_high, long tenure, prior unresolved, high frustration)
- churn_risk: true only if customer mentions switching, porting, or a specific competitor
- Do NOT output RESOLVE on the greeting turn or before understanding the issue
- Output RESOLVE exactly once per action (raise, confirm, escalate, or close)
"""

RESOLVE_RE = re.compile(r'RESOLVE:(\{[^}]+\})', re.DOTALL)
_FRUSTRATION_TAG_RE = re.compile(r'\[Frustration:\s*\w+\]\s*', re.IGNORECASE)


class ConversationAgent:
    def __init__(self, session_id: str, customer_id: str, language_code: str, language_name: str):
        self.session_id    = session_id
        self.customer_id   = customer_id
        self.language_code = language_code
        self.language_name = language_name

        self.customer_profile      = None
        self.tickets               = []
        self.prior_unresolved      = 0

        self.chat                  = None
        self.messages              = []   # [{role: "agent"|"user", text: str}]
        self.is_resolved           = False
        self.resolution            = None
        self.ticket_raised         = False
        self.raised_ticket_id      = None
        self.consecutive_frustrated = 0   # turns with detected high/critical frustration

    async def initialize(self) -> str:
        """Load customer context, start Gemini chat, return greeting text."""
        self.customer_profile, self.tickets = await asyncio.gather(
            get_customer_profile(self.customer_id),
            get_tickets(self.customer_id),
        )
        self.prior_unresolved = sum(
            1 for t in self.tickets if t["status"] in ("closed_unresolved", "auto_closed")
        )
        open_tickets = [t for t in self.tickets if t["status"] == "open"]

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            language_name=self.language_name,
            first_name=self.customer_profile["name"].split()[0],
            plan_name=self.customer_profile["plan_name"],
            arpu_monthly=self.customer_profile["arpu_monthly"],
            arpu_tier=self.customer_profile["arpu_tier"],
            tenure_years=self.customer_profile["tenure_years"],
            prior_unresolved=self.prior_unresolved,
            open_tickets=len(open_tickets),
        )

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            system_instruction=system_prompt,
        )
        self.chat = model.start_chat(history=[])

        resp     = await self.chat.send_message_async("START")
        greeting = self._extract_spoken(resp.text)
        self.messages.append({"role": "agent", "text": greeting})
        return greeting

    async def send_message(self, user_text: str, audio_emotion: dict | None = None) -> dict:
        """Process one user turn. Returns {text, is_resolved, branch, ticket_id, handoff_brief, decision}."""
        self.messages.append({"role": "user", "text": user_text})

        # Combine text-keyword detection with SER (voice) emotion — take the higher of the two
        frustration_level = self._detect_frustration(user_text, audio_emotion)
        tagged_input = f"[Frustration: {frustration_level}] {user_text}"

        resp   = await self.chat.send_message_async(tagged_input)
        raw    = resp.text
        match  = RESOLVE_RE.search(raw)
        spoken = self._extract_spoken(raw)

        if match:
            try:
                decision = json.loads(match.group(1))
            except json.JSONDecodeError:
                decision = {
                    "intent": "Network or service issue",
                    "frustration_level": frustration_level,
                    "recommended_action": "raise_ticket",
                    "priority_score": 5,
                    "churn_risk": False,
                    "frustration_signals": [],
                }
            return await self._close(decision, spoken_text=spoken)

        self.messages.append({"role": "agent", "text": spoken})
        return {
            "text":          spoken,
            "is_resolved":   False,
            "branch":        None,
            "ticket_id":     self.raised_ticket_id,
            "handoff_brief": None,
            "decision":      None,
            "frustration":   frustration_level,
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    _FRUSTRATION_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    def _detect_frustration(self, text: str, audio_emotion: dict | None = None) -> str:
        """
        Both text-keyword and SER work independently — either one can flag frustration.
        The higher of the two signals wins.

        - Text: triggers on explicit frustration words (useless, worst, fraud, port…)
        - SER:  triggers on vocal tone (anger, disgust, fear) — only when confidence
                clears the per-level gate in ser.py, so weak detections stay "low"
        """
        order      = self._FRUSTRATION_ORDER
        text_level = self._text_frustration(text)
        ser_level  = (audio_emotion or {}).get("frustration_level", "low")
        effective  = text_level if order.get(text_level, 0) >= order.get(ser_level, 0) else ser_level

        if order.get(effective, 0) >= 2:      # high or critical
            self.consecutive_frustrated += 1
        elif order.get(effective, 0) == 0:    # low
            self.consecutive_frustrated = max(0, self.consecutive_frustrated - 1)

        return effective

    def _text_frustration(self, text: str) -> str:
        """Keyword-only scan on transcript text."""
        t = text.lower()
        if any(w in t for w in _CRITICAL_FRUSTRATION):
            return "critical"
        if any(w in t for w in _HIGH_FRUSTRATION):
            return "high"
        if t.count("?") >= 2 or "please" in t:
            return "medium"
        return "low"

    def _extract_spoken(self, raw: str) -> str:
        text = RESOLVE_RE.sub("", raw)
        text = _FRUSTRATION_TAG_RE.sub("", text)
        return text.strip()

    async def _quick_reply(self, prompt: str, fallback: str) -> str:
        """Short Gemini call with tight token limit and a fallback on any error."""
        try:
            resp = await self.chat.send_message_async(
                prompt,
                generation_config=genai.types.GenerationConfig(max_output_tokens=80),
            )
            text = self._extract_spoken(resp.text).strip()
            return text if text else fallback
        except Exception:
            return fallback

    async def _close(self, decision: dict, spoken_text: str | None) -> dict:
        profile        = self.customer_profile
        open_ticket    = next((t for t in self.tickets if t["status"] == "open"), None)
        action         = decision.get("recommended_action", "raise_ticket")
        frustration    = decision.get("frustration_level", "medium")
        churn_risk     = decision.get("churn_risk", False)
        priority_score = decision.get("priority_score", 5)

        # ── Tier-aware escalation override ───────────────────────────────
        arpu_tier  = profile.get("arpu_tier", "prepaid_monthly")
        is_premium = arpu_tier in ("black", "postpaid_high")

        if churn_risk:
            # Churn risk always escalates regardless of tier
            action = "escalate_human"
        elif is_premium:
            # Premium: human support at medium frustration or above
            if (frustration in ("medium", "high", "critical") or
                    self.consecutive_frustrated >= 2 or
                    (self.prior_unresolved >= 2 and frustration != "low")):
                action = "escalate_human"
        else:
            # Standard / low ARPU: ticket + assurance first;
            # escalate only when clearly agitated (high/critical) or very persistent
            if (frustration in ("high", "critical") or
                    self.consecutive_frustrated >= 4 or
                    (self.prior_unresolved >= 2 and frustration in ("high", "critical"))):
                action = "escalate_human"
            elif action == "escalate_human":
                # Gemini wanted to escalate but threshold not met — downgrade to ticket
                action = "raise_ticket"

        if action == "update_existing" and not open_ticket:
            action = "raise_ticket"

        # ── Confirm existing ticket ───────────────────────────────────────
        if action == "update_existing":
            inj = f"Ticket {open_ticket['ticket_id']} confirmed open. 1-2 sentences: inform customer, ask if anything else. No RESOLVE."
            follow_up = await self._quick_reply(inj,
                fallback=f"Your ticket {open_ticket['ticket_id']} is already open and our team is actively working on it — is there anything else I can help you with today?"
            )
            self.messages.append({"role": "agent", "text": follow_up})
            return {
                "text":          follow_up,
                "is_resolved":   False,
                "branch":        "ticket_exists",
                "ticket_id":     open_ticket["ticket_id"],
                "handoff_brief": None,
                "decision":      decision,
                "priority_score": priority_score,
                "priority_tier":  compute_priority_tier(priority_score),
                "frustration":   frustration,
            }

        # ── Escalate to human ─────────────────────────────────────────────
        elif action == "escalate_human":
            tier = compute_priority_tier(priority_score)
            convo_summary = "\n".join(
                f"{'Customer' if m['role'] == 'user' else 'Aria'}: {m['text']}"
                for m in self.messages
            )
            handoff_brief = {
                "customer_name":        profile["name"],
                "mobile":               profile["mobile"],
                "arpu_tier":            profile["arpu_tier"],
                "plan":                 profile["plan_name"],
                "tenure_years":         profile["tenure_years"],
                "priority_tier":        tier,
                "priority_score":       priority_score,
                "parsed_issue":         decision.get("intent", ""),
                "frustration_level":    frustration,
                "frustration_signals":  decision.get("frustration_signals", []),
                "churn_risk":           churn_risk,
                "prior_unresolved":     self.prior_unresolved,
                "ticket_history":       self.tickets,
                "raised_ticket_id":     self.raised_ticket_id,
                "first_move":           self._first_move(decision),
                "conversation_summary": convo_summary,
            }
            if churn_risk:
                response_text = spoken_text or (
                    "I completely understand your frustration and I sincerely apologise. "
                    "I'm connecting you right now with a senior specialist who has your full history "
                    "and the authority to resolve this immediately — you will not need to repeat yourself."
                )
            elif frustration == "critical" or self.consecutive_frustrated >= 3:
                response_text = spoken_text or (
                    "I hear you, and I want to make sure this gets resolved properly. "
                    "I'm escalating this to a specialist who already has everything you've told me."
                )
            else:
                response_text = spoken_text or (
                    "I'm escalating this to our specialist team who will call you back shortly. "
                    "They already have your full details and complaint history."
                )
            self.messages.append({"role": "agent", "text": response_text})
            self.is_resolved = True
            self.resolution  = {"branch": "escalate", "decision": decision, "response_text": response_text}
            return {
                "text":          response_text,
                "is_resolved":   True,
                "branch":        "escalate",
                "ticket_id":     self.raised_ticket_id,
                "handoff_brief": handoff_brief,
                "decision":      decision,
                "priority_score": priority_score,
                "priority_tier":  tier,
                "frustration":   frustration,
            }

        # ── Customer satisfied — close ────────────────────────────────────
        elif action == "close":
            response_text = spoken_text or (
                "I'm glad I could help. Thank you for contacting Airtel — have a wonderful day!"
            )
            self.messages.append({"role": "agent", "text": response_text})
            self.is_resolved = True
            self.resolution  = {"branch": "close", "decision": decision, "response_text": response_text}
            return {
                "text":          response_text,
                "is_resolved":   True,
                "branch":        "close",
                "ticket_id":     self.raised_ticket_id,
                "handoff_brief": None,
                "decision":      decision,
                "priority_score": priority_score,
                "priority_tier":  compute_priority_tier(priority_score),
                "frustration":   frustration,
            }

        # ── Raise new ticket, then continue conversation ──────────────────
        else:
            ticket_id = await crm_raise_ticket(
                customer_id=self.customer_id,
                issue=decision.get("intent", "Customer complaint"),
                priority=priority_score,
            )
            await send_sms(
                profile["mobile"],
                f"Your complaint has been registered. Ticket: {ticket_id}. "
                "Our team will contact you within 24 hours.",
            )
            await store_conversation(
                self.customer_id,
                " | ".join(m["text"] for m in self.messages if m["role"] == "user"),
                decision.get("intent", ""),
            )
            self.ticket_raised    = True
            self.raised_ticket_id = ticket_id

            if is_premium:
                inj = f"Ticket {ticket_id} raised, SMS sent. 1-2 sentences: inform customer of ticket {ticket_id}, ask if anything else. No RESOLVE."
                fb  = f"Your ticket {ticket_id} has been raised and an SMS confirmation has been sent — is there anything else I can help you with today?"
            else:
                inj = f"Ticket {ticket_id} raised, SMS sent. 1-2 sentences: reassure the customer their issue is a priority and WILL be resolved, give ticket {ticket_id}, ask if anything else. Warm and confident tone. No RESOLVE."
                fb  = f"I want to assure you that your concern is important to us and will be resolved — your ticket {ticket_id} is raised and our team will be in touch within 24 hours. Is there anything else I can help you with?"
            follow_up = await self._quick_reply(inj, fallback=fb)
            self.messages.append({"role": "agent", "text": follow_up})
            return {
                "text":          follow_up,
                "is_resolved":   False,
                "branch":        "raise_ticket",
                "ticket_id":     ticket_id,
                "handoff_brief": None,
                "decision":      decision,
                "priority_score": priority_score,
                "priority_tier":  compute_priority_tier(priority_score),
                "frustration":   frustration,
            }

    def _first_move(self, decision: dict) -> str:
        if decision.get("churn_risk"):
            return "CHURN RISK — Do NOT use a script. Retention offer pre-authorised. Last chance contact."
        if self.prior_unresolved >= 2:
            return (f"Customer has {self.prior_unresolved} prior unresolved tickets. "
                    "Do NOT ask them to explain again. Acknowledge history immediately.")
        return (f"Issue: {decision.get('intent')}. "
                f"Tone: {decision.get('frustration_level')}. Proceed with standard resolution.")
