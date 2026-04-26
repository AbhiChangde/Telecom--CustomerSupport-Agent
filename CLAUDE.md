# TeleSupport AI — Claude Code Project Brief

> This file is the single source of truth for building the TeleSupport AI agent.
> Read it fully before writing any code. Every architectural decision is documented here with reasoning.

---

## What you are building

An intelligent customer support agent for Indian telecom operators (modelled on Airtel).
It is a **thesis prototype** — correctness of architecture matters more than production hardening.

The agent is a **6-state LangGraph pipeline** that:
1. Lets the customer choose their language
2. Plays a pre-cached greeting (no LLM cost)
3. Accepts the customer's problem as a voice note (Sarvam STT) or typed text
4. Fetches customer context in parallel (CRM + Vector DB)
5. Runs one Gemini Flash call to understand intent, frustration, and priority
6. Routes to one of three resolution branches: raise ticket / ticket exists / escalate to human

The agent does **not** resolve network or billing issues. It eliminates friction in the support process — zero re-identification, accurate intent understanding in Hinglish/Indian languages, and a structured handoff brief for the human agent.

---

## Repository structure to create

```
telesupport-ai/
├── CLAUDE.md                  # This file
├── README.md
├── requirements.txt
├── .env.example
│
├── main.py                    # FastAPI app entrypoint
│
├── agent/
│   ├── __init__.py
│   ├── graph.py               # LangGraph graph definition — all nodes and edges
│   ├── state.py               # SupportAgentState TypedDict
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── language_node.py   # State 1
│   │   ├── greeting_node.py   # State 2
│   │   ├── input_node.py      # State 3
│   │   ├── context_node.py    # State 4 — parallel fetch
│   │   ├── decision_node.py   # State 5 — Gemini call
│   │   └── resolution_node.py # State 6 — three branches
│   └── conditions.py          # Routing functions and frustration override
│
├── services/
│   ├── __init__.py
│   ├── sarvam.py              # Sarvam STT integration
│   ├── gemini.py              # Gemini Flash LLM wrapper
│   ├── vector_db.py           # Vector DB client (Pinecone or Weaviate)
│   ├── crm.py                 # CRM API mock (returns realistic fake data)
│   └── sms.py                 # SMS confirmation mock
│
├── models/
│   ├── __init__.py
│   ├── customer.py            # Customer profile Pydantic models
│   ├── ticket.py              # Ticket Pydantic models
│   └── decision.py            # Gemini response Pydantic model
│
├── api/
│   ├── __init__.py
│   ├── routes.py              # FastAPI route definitions
│   └── schemas.py             # Request/response schemas
│
├── data/
│   ├── greetings/             # Pre-cached greeting audio files (placeholder)
│   │   ├── hi.mp3
│   │   ├── en.mp3
│   │   ├── ta.mp3
│   │   ├── te.mp3
│   │   ├── mr.mp3
│   │   └── bn.mp3
│   └── mock/
│       ├── customers.json     # Mock customer profiles for demo
│       └── tickets.json       # Mock ticket history for demo
│
└── tests/
    ├── __init__.py
    ├── test_graph.py
    ├── test_decision.py
    └── test_routing.py
```

---

## Tech stack — exact versions

```
python = ">=3.11"
fastapi = ">=0.111"
uvicorn = ">=0.29"
langgraph = ">=0.1"
langchain-google-genai = ">=1.0"    # Gemini Flash
httpx = ">=0.27"                     # async HTTP for Sarvam
python-multipart = ">=0.0.9"        # file upload for audio
pydantic = ">=2.0"
pinecone-client = ">=3.0"           # OR weaviate-client>=4.0
python-dotenv = ">=1.0"
```

---

## Environment variables (.env.example)

```
GEMINI_API_KEY=your_gemini_api_key
SARVAM_API_KEY=your_sarvam_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=telesupport-memory
VECTOR_DB_PROVIDER=pinecone          # or weaviate
USE_MOCK_CRM=true                    # true for demo, false for production
USE_MOCK_VECTOR_DB=true              # true for demo
USE_MOCK_SMS=true
```

---

## State schema — build this first

File: `agent/state.py`

```python
from typing import TypedDict, Optional, Literal

class CustomerProfile(TypedDict):
    customer_id: str
    name: str
    mobile: str
    arpu_monthly: float          # e.g. 1499.0
    arpu_tier: Literal["black", "postpaid_high", "postpaid_std", "prepaid_annual", "prepaid_monthly"]
    tenure_years: float          # e.g. 5.5
    plan_type: Literal["postpaid", "prepaid"]
    plan_name: str               # e.g. "Airtel Black 1499"
    location: str

class Ticket(TypedDict):
    ticket_id: str
    issue_summary: str
    status: Literal["open", "closed_resolved", "closed_unresolved", "auto_closed"]
    created_date: str
    closed_date: Optional[str]
    issue_category: str

class GeminiDecision(TypedDict):
    intent: str                  # one sentence
    frustration_level: Literal["low", "medium", "high", "critical"]
    frustration_signals: list[str]
    ticket_status: Literal["none", "open", "closed_unresolved"]
    recommended_action: Literal["raise_ticket", "update_existing", "escalate_human"]
    priority_score: int          # 1-10
    priority_reasoning: str
    churn_risk: bool

class SupportAgentState(TypedDict):
    # State 1 — Language
    customer_id: str
    language_code: str           # hi, en, ta, te, mr, bn
    language_name: str

    # State 2 — Greeting (no new fields, greeting played via API response)

    # State 3 — Input
    user_input: str              # transcript or typed text
    input_mode: Literal["voice", "text"]
    audio_duration_seconds: Optional[float]

    # State 4 — Context
    customer_profile: Optional[CustomerProfile]
    tickets: list[Ticket]        # all tickets for this customer
    prior_ticket_count: int      # tickets on same issue
    vector_memory: list[dict]    # similar past complaints from vector DB
    outage_status: str           # "none" or description

    # State 5 — Decision
    decision: Optional[GeminiDecision]
    is_frustrated: bool          # computed override condition

    # State 6 — Resolution
    response_text: str           # what the agent says back
    new_ticket_id: Optional[str]
    handoff_brief: Optional[dict]
    resolution_branch: Optional[Literal["raise_ticket", "ticket_exists", "escalate"]]
```

---

## LangGraph graph — build this second

File: `agent/graph.py`

```python
from langgraph.graph import StateGraph, END
from agent.state import SupportAgentState
from agent.nodes import (
    language_node, greeting_node, input_node,
    context_node, decision_node, resolution_node
)
from agent.conditions import route_resolution, check_frustration_override

def build_graph() -> StateGraph:
    graph = StateGraph(SupportAgentState)

    graph.add_node("language", language_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("input", input_node)
    graph.add_node("context", context_node)
    graph.add_node("decision", decision_node)
    graph.add_node("raise_ticket", resolution_node.raise_ticket)
    graph.add_node("ticket_exists", resolution_node.ticket_exists)
    graph.add_node("escalate", resolution_node.escalate)

    graph.set_entry_point("language")
    graph.add_edge("language", "greeting")
    graph.add_edge("greeting", "input")
    graph.add_edge("input", "context")
    graph.add_edge("context", "decision")

    # After decision node: check frustration override first, then route
    graph.add_conditional_edges(
        "decision",
        route_resolution,
        {
            "raise_ticket": "raise_ticket",
            "ticket_exists": "ticket_exists",
            "escalate": "escalate",
        }
    )

    graph.add_edge("raise_ticket", END)
    graph.add_edge("ticket_exists", END)
    graph.add_edge("escalate", END)

    return graph.compile()

# Singleton
support_graph = build_graph()
```

---

## Routing and frustration override

File: `agent/conditions.py`

```python
from agent.state import SupportAgentState

CHURN_WORDS = [
    "port", "jio", "vi", "bsnl", "vodafone", "airtel chhod",
    "consumer forum", "trai", "legal action", "nch", "done with",
    "switching", "never using", "last time", "bye airtel"
]

def is_critically_frustrated(state: SupportAgentState) -> bool:
    decision = state.get("decision", {})
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
    # Frustration override takes priority over everything
    if is_critically_frustrated(state):
        return "escalate"

    action = state.get("decision", {}).get("recommended_action", "raise_ticket")

    if action == "escalate_human":
        return "escalate"
    elif action == "update_existing":
        return "ticket_exists"
    else:
        return "raise_ticket"
```

---

## Node implementations

### State 4 — context_node.py (most important node)

```python
import asyncio
from agent.state import SupportAgentState
from services.crm import get_customer_profile, get_tickets
from services.vector_db import recall_similar_complaints
from services.outage import check_outage  # mock

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

    # Count tickets on the SAME issue (semantic match)
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
```

### State 5 — decision_node.py

```python
import json
from agent.state import SupportAgentState
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

Return this exact JSON structure:
{{
  "intent": "one sentence describing the core issue",
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

async def decision_node(state: SupportAgentState) -> SupportAgentState:
    profile = state["customer_profile"]
    tickets = state["tickets"]
    open_tickets = [t for t in tickets if t["status"] == "open"]
    vector_summary = f"{len(state['vector_memory'])} similar past complaints found" if state["vector_memory"] else "none"

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
    decision = json.loads(raw)

    return {**state, "decision": decision}
```

### State 6 — resolution_node.py

```python
from agent.state import SupportAgentState
from agent.conditions import is_critically_frustrated, compute_priority_tier
from services.crm import raise_ticket, get_ticket_status
from services.sms import send_sms
from services.vector_db import store_conversation

async def raise_ticket_branch(state: SupportAgentState) -> SupportAgentState:
    profile = state["customer_profile"]
    decision = state["decision"]

    ticket_id = await raise_ticket(
        customer_id=state["customer_id"],
        issue=decision["intent"],
        priority=decision["priority_score"],
    )
    await send_sms(profile["mobile"], f"Your complaint has been registered. Ticket: {ticket_id}. Our team will contact you within 24 hours.")
    await store_conversation(state["customer_id"], state["user_input"], decision["intent"])

    response = f"I've raised a priority ticket for your {decision['intent']}. Ticket ID: {ticket_id}. You'll receive an SMS confirmation. Our team will reach out within 24 hours."

    return {**state, "response_text": response, "new_ticket_id": ticket_id, "resolution_branch": "raise_ticket"}

async def ticket_exists_branch(state: SupportAgentState) -> SupportAgentState:
    open_ticket = next((t for t in state["tickets"] if t["status"] == "open"), None)
    response = f"I can see you already have an open ticket ({open_ticket['ticket_id']}) for this issue raised on {open_ticket['created_date']}. Our team is actively working on it. You'll be contacted within the committed timeline."

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
        "first_move": generate_first_move(decision, state["tickets"]),
    }

    if decision["churn_risk"]:
        response = "I completely understand your frustration. I'm connecting you with a senior specialist right now who has your full history and the authority to resolve this immediately. You will not need to explain anything again."
    else:
        response = "I'm escalating this to our support team who will call you back shortly. They already have your full details and complaint history."

    return {**state, "response_text": response, "handoff_brief": brief, "resolution_branch": "escalate"}

def generate_first_move(decision: dict, tickets: list) -> str:
    if decision["churn_risk"]:
        return "CHURN RISK — Do NOT use a script. Acknowledge complaint history by date. Retention offer pre-authorised. Last chance contact."
    if decision["prior_ticket_count"] >= 2:
        return f"Customer has {decision['prior_ticket_count']} unresolved tickets on this issue. Do NOT ask them to explain again. Acknowledge history immediately."
    return f"Issue: {decision['intent']}. Tone is {decision['frustration_level']}. Proceed with standard resolution but acknowledge wait time."

# Export branch map for graph wiring
raise_ticket = raise_ticket_branch
ticket_exists = ticket_exists_branch
escalate = escalate_branch
```

---

## Sarvam STT integration

File: `services/sarvam.py`

```python
import httpx
import os
from fastapi import UploadFile

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"

async def transcribe(audio_bytes: bytes, language_code: str) -> str:
    """
    Transcribe audio using Sarvam saaras:v2.
    language_code: hi, en, ta, te, mr, bn — or 'unknown' for auto-detect.
    Returns transcript string.
    """
    if os.getenv("USE_MOCK_STT", "false").lower() == "true":
        return "[MOCK TRANSCRIPT] My 5G is showing but data is still getting used up. This has been happening for 3 days."

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            SARVAM_STT_URL,
            headers={"api-subscription-key": SARVAM_API_KEY},
            files={"file": ("audio.webm", audio_bytes, "audio/webm")},
            data={
                "model": "saaras:v2",
                "language_code": language_code if language_code != "en" else "unknown",
                "with_timestamps": "false",
            }
        )
    response.raise_for_status()
    return response.json()["transcript"]
```

---

## Gemini Flash integration

File: `services/gemini.py`

```python
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

async def call_gemini(prompt: str) -> str:
    """
    Single Gemini Flash call. Returns raw text response.
    Always call with JSON-only instruction in the prompt.
    """
    response = await model.generate_content_async(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,      # low temperature for consistent JSON
            max_output_tokens=800,
        )
    )
    return response.text.strip()
```

---

## Mock CRM data

File: `data/mock/customers.json`

```json
{
  "CUST001": {
    "customer_id": "CUST001",
    "name": "Rahul Verma",
    "mobile": "+919876543210",
    "arpu_monthly": 1499.0,
    "arpu_tier": "black",
    "tenure_years": 5.2,
    "plan_type": "postpaid",
    "plan_name": "Airtel Black 1499",
    "location": "Hyderabad"
  },
  "CUST002": {
    "customer_id": "CUST002",
    "name": "Priya Sharma",
    "mobile": "+917788812345",
    "arpu_monthly": 299.0,
    "arpu_tier": "prepaid_monthly",
    "tenure_years": 0.4,
    "plan_type": "prepaid",
    "plan_name": "Prepaid 299",
    "location": "Mumbai"
  },
  "CUST003": {
    "customer_id": "CUST003",
    "name": "Suresh Iyer",
    "mobile": "+919012345678",
    "arpu_monthly": 999.0,
    "arpu_tier": "postpaid_high",
    "tenure_years": 11.0,
    "plan_type": "postpaid",
    "plan_name": "Postpaid 999",
    "location": "Chennai"
  },
  "CUST004": {
    "customer_id": "CUST004",
    "name": "Deepak Nair",
    "mobile": "+918823456789",
    "arpu_monthly": 549.0,
    "arpu_tier": "postpaid_std",
    "tenure_years": 3.1,
    "plan_type": "postpaid",
    "plan_name": "Postpaid 549",
    "location": "Bangalore"
  }
}
```

---

## FastAPI routes

File: `api/routes.py`

```python
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from agent.graph import support_graph
from agent.state import SupportAgentState
from services.sarvam import transcribe
import uuid

router = APIRouter()

@router.post("/session/start")
async def start_session(customer_id: str, language_code: str):
    """State 1 + 2: Initialise session and return greeting audio URL."""
    greeting_url = f"/static/greetings/{language_code}.mp3"
    return {
        "session_id": str(uuid.uuid4()),
        "customer_id": customer_id,
        "language_code": language_code,
        "greeting_url": greeting_url,
    }

@router.post("/message/voice")
async def process_voice(
    customer_id: str = Form(...),
    language_code: str = Form(...),
    audio: UploadFile = File(...),
):
    """State 3 (voice path): Transcribe audio then run full pipeline."""
    audio_bytes = await audio.read()
    transcript = await transcribe(audio_bytes, language_code)

    initial_state = _build_initial_state(customer_id, language_code, transcript, "voice")
    result = await support_graph.ainvoke(initial_state)

    return _format_response(result, transcript)

@router.post("/message/text")
async def process_text(customer_id: str, language_code: str, text: str):
    """State 3 (text path): Run full pipeline with typed input."""
    initial_state = _build_initial_state(customer_id, language_code, text, "text")
    result = await support_graph.ainvoke(initial_state)

    return _format_response(result)

def _build_initial_state(customer_id, language_code, user_input, input_mode) -> SupportAgentState:
    language_names = {"hi":"Hindi","en":"English","ta":"Tamil","te":"Telugu","mr":"Marathi","bn":"Bengali"}
    return SupportAgentState(
        customer_id=customer_id,
        language_code=language_code,
        language_name=language_names.get(language_code, "English"),
        user_input=user_input,
        input_mode=input_mode,
        audio_duration_seconds=None,
        customer_profile=None,
        tickets=[],
        prior_ticket_count=0,
        vector_memory=[],
        outage_status="none",
        decision=None,
        is_frustrated=False,
        response_text="",
        new_ticket_id=None,
        handoff_brief=None,
        resolution_branch=None,
    )

def _format_response(result: SupportAgentState, transcript: str = None) -> dict:
    response = {
        "response": result["response_text"],
        "resolution_branch": result["resolution_branch"],
        "priority_score": result["decision"]["priority_score"] if result["decision"] else None,
        "priority_tier": _get_tier(result["decision"]["priority_score"]) if result["decision"] else None,
    }
    if transcript:
        response["transcript"] = transcript
    if result["new_ticket_id"]:
        response["ticket_id"] = result["new_ticket_id"]
    if result["handoff_brief"]:
        response["handoff_brief"] = result["handoff_brief"]
    return response

def _get_tier(score: int) -> str:
    if score >= 8: return "P1"
    if score >= 5: return "P2"
    return "P3"
```

---

## main.py

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="TeleSupport AI",
    description="Intelligent telecom customer support agent — thesis prototype",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="data"), name="static")
app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return {"status": "ok", "message": "TeleSupport AI is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

---

## Priority scoring helper

File: `agent/conditions.py` — add this function:

```python
def compute_priority_tier(score: int) -> str:
    if score >= 8: return "P1"
    if score >= 5: return "P2"
    return "P3"

def compute_priority_score(
    arpu_tier: str,
    tenure_years: float,
    prior_unresolved: int,
    frustration_level: str,
) -> int:
    """
    Deterministic fallback scorer if Gemini returns invalid score.
    Mirrors the prompt scoring guide exactly.
    """
    arpu_scores = {"black":10,"postpaid_high":8,"postpaid_std":6,"prepaid_annual":4,"prepaid_monthly":2}
    arpu_score = arpu_scores.get(arpu_tier, 4)

    tenure_score = 10 if tenure_years >= 5 else 6 if tenure_years >= 2 else 2

    prior_score = 10 if prior_unresolved >= 3 else 7 if prior_unresolved == 2 else 4 if prior_unresolved == 1 else 1

    frustration_scores = {"critical":10,"high":7,"medium":4,"low":1}
    frustration_score = frustration_scores.get(frustration_level, 1)

    weighted = (arpu_score * 0.35) + (tenure_score * 0.20) + (prior_score * 0.30) + (frustration_score * 0.15)
    return min(10, max(1, round(weighted)))
```

---

## Build order

Follow this sequence to avoid circular imports and broken tests at each stage:

1. `agent/state.py` — TypedDict definitions. No dependencies.
2. `models/` — Pydantic models for CRM, ticket, decision.
3. `data/mock/` — JSON files for customers and tickets.
4. `services/crm.py` — reads mock JSON, returns CustomerProfile and Ticket lists.
5. `services/gemini.py` — Gemini Flash wrapper. Test standalone with a simple prompt.
6. `services/sarvam.py` — STT wrapper. Set `USE_MOCK_STT=true` initially.
7. `services/vector_db.py` — mock implementation that returns empty list initially.
8. `services/sms.py` — mock that prints to stdout.
9. `agent/conditions.py` — routing functions and priority scorer.
10. `agent/nodes/` — one file at a time, test each node in isolation before wiring.
11. `agent/graph.py` — wire all nodes. Run `support_graph.invoke()` with a test state.
12. `api/routes.py` + `main.py` — FastAPI wrapper. Test with curl or httpx.
13. `tests/` — write tests for decision routing and frustration override.

---

## Key architectural decisions and why

**Why LangGraph over plain Python:**
Thesis project needs formal, citable architecture. LangGraph gives: conditional edge routing, global frustration override at any state, LangSmith tracing for evaluation metrics, self-documenting graph that renders as a figure in the thesis.

**Why Sarvam over Whisper or ElevenLabs for STT:**
Sarvam saaras:v2 is purpose-built for Indian languages and Hinglish code-switching. Outperforms GPT-4o Transcribe on IndicVoices benchmark. Critical for this use case — customers speak in Hinglish mixes like "mera 5G chal nahi raha last 3 days se".

**Why Gemini Flash over GPT-4o:**
Cheaper (~$0.075/million input tokens vs ~$0.50), natively multilingual including Hindi, sufficient capability for intent classification and frustration scoring. Reserve GPT-4o for future work if needed.

**Why async throughout:**
State 4 fires three parallel API calls using asyncio.gather. Sequential calls would add 1-2 seconds of latency. FastAPI is async-native. LangGraph supports async nodes with ainvoke.

**Why mock CRM for thesis:**
The thesis contribution is the agent architecture, not the CRM integration. Mock data covers all four demo scenarios (high ARPU, new customer, long tenure, churn risk). Real CRM integration is documented as future work.

**Why voice note (async) over real-time voice agent:**
Indian users send 22% of WhatsApp communications as voice notes vs 14% global average. Async voice eliminates real-time pressure, works on slow connections, and avoids ElevenLabs TTS cost on every agent response. TTS is available as opt-in, not default.

**Why text-only agent response by default:**
ElevenLabs TTS costs ~$0.18 per conversation vs $0.0003 for Gemini Flash text. At scale, TTS makes the system 600x more expensive. Text response is shown in chat UI; TTS is offered as an accessibility option.

---

## Evaluation dataset

The file `data/airtel_support_cluster_reviews.xlsx` (if included) contains 288 manually labelled reviews from the Airtel Thanks app (March 2026) with the following columns:
- `review_id`, `rating`, `review_text`, `issue_categories`
- Manual label to add: `expected_action` (raise_ticket / update_existing / escalate_human)

Use this for Metric 1 (F1 score) in the thesis evaluation chapter. Run all 288 through the agent and compare `resolution_branch` against `expected_action`.

---

## What NOT to build (scope boundaries)

- Do not build a real-time voice AI agent (real-time two-way voice call). This is async voice note only.
- Do not integrate a real CRM. Mock data is sufficient for thesis.
- Do not build a WhatsApp integration. The demo UI (telecom_support_prototype.html) is the frontend.
- Do not build a retention offer engine. The handoff brief flags churn risk and pre-authorises offers — human agent executes them.
- Do not build multi-agent orchestration. Single LangGraph pipeline with one Gemini call per conversation.

---

## Running the project

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill environment variables
cp .env.example .env

# Run with all mocks enabled (no API keys needed for initial build)
USE_MOCK_CRM=true USE_MOCK_STT=true USE_MOCK_VECTOR_DB=true USE_MOCK_SMS=true uvicorn main:app --reload

# Test a text message
curl -X POST "http://localhost:8000/api/message/text" \
  -d "customer_id=CUST001&language_code=en&text=My 5G is not working for 3 days"

# API docs
open http://localhost:8000/docs
```

---

*End of CLAUDE.md — begin building from agent/state.py*
