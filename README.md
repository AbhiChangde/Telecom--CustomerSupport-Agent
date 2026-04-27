# TeleSupport AI

An intelligent, multi-turn voice and text customer support agent for Indian telecom operators, modelled on Airtel. Built as a thesis prototype demonstrating how conversational AI can eliminate friction in telecom support — from re-identification and language barriers to unstructured escalations and missed context.

---

## The Problem

Customer support in Indian telecom is broken in predictable ways:

- **Re-identification on every call.** Customers repeat their name, number, and issue to each agent they reach, even when they have called before.
- **Language mismatch.** IVR systems and agents often cannot handle Hinglish, Tamil, or Telugu code-switching that customers naturally use.
- **Frustrated customers get generic responses.** High-value customers who are already agitated receive the same scripted reply as a first-time caller.
- **No structured handoff.** When a human agent finally takes the call, they receive no context — the customer has to start from scratch.
- **Recurring issues go unnoticed.** When the same customer raises the same issue across multiple sessions, no agent knows — each session looks like the first time.

TeleSupport AI addresses each of these by loading customer context before the first word is spoken, tracking issue history across sessions, adapting tone and routing based on frustration and ARPU tier, and handing off a detailed brief to the human agent if escalation is needed.

---

## What It Does

The agent is named **Aria**. She conducts a short, empathetic spoken or typed conversation with the customer, then routes the interaction to one of three outcomes:

| Outcome | When |
|---|---|
| **Raise ticket** | New issue not already tracked |
| **Confirm existing ticket** | Open ticket already exists for this issue |
| **Escalate to human** | Premium customer with frustration, recurring issue pattern, churn risk, or any customer who is highly agitated |

On escalation, the human agent receives a structured **handoff brief** — customer profile, ARPU tier, full conversation transcript, frustration signals, prior unresolved count, similar past issues, and a suggested first move — so they never have to ask the customer to repeat themselves.

---

## Key Features

### Conversational, multi-turn flow
Aria holds a real multi-turn conversation via a persistent Gemini 2.5 Flash chat session. She greets, empathises, clarifies if needed, resolves, follows up ("Is there anything else I can help with?"), and only closes when the customer is satisfied or escalation is triggered.

### Per-turn frustration detection (dual-signal)
Every customer message is evaluated by two independent signals before it reaches Gemini:

| Signal | Source | How it works |
|---|---|---|
| **Text keywords** | Transcript or typed text | Scans for frustration words, expletives, insults, and churn signals — also covers Hindi (`बेकार`, `बेवकूफ`, `निकम्मा`) |
| **Voice emotion (SER)** | Raw audio only | `superb/wav2vec2-base-superb-er` classifies vocal tone into emotion categories and maps them to frustration levels |

The higher of the two signals wins. A calm sentence spoken in an angry tone, or an aggressive sentence spoken quietly, will each be caught by the appropriate signal.

**Text signal coverage:**

| Category | Examples |
|---|---|
| High frustration | `useless`, `terrible`, `fraud`, `scam`, `never works` |
| Expletives / insults | `stupid`, `idiot`, `bullshit`, `wtf`, `you suck`, `are you serious` |
| Hindi insults | `बेवकूफ`, `गधा`, `मूर्ख`, `निकम्मा`, `फालतू` |
| Critical / churn | `port`, `jio`, `vi`, `legal action`, `consumer forum`, `switching` |

Each message is tagged `[Frustration: low/medium/high/critical]` before being sent to Gemini, so the LLM adapts its tone and routing on every turn — not just at resolution time. A consecutive frustrated turn counter tracks sustained agitation across multiple turns.

### ARPU-tier-aware routing
Escalation thresholds differ by customer value:

| Tier | Escalation trigger |
|---|---|
| `black` / `postpaid_high` | Medium frustration or above, OR 2+ similar past issues |
| `postpaid_std` / `prepaid_*` | High or critical frustration only |

Standard customers who are mildly frustrated receive warm ticket assurance. Even when Gemini recommends escalation for a low-ARPU customer at medium frustration, the backend overrides it to raise a ticket instead.

### Cross-session memory (ChromaDB)
Every resolved issue is stored in a local ChromaDB vector store using `all-MiniLM-L6-v2` embeddings. On every new session:

- All past issues for the customer are loaded at startup and injected into Aria's system prompt — she is aware of the customer's history before the first message
- When a resolution is reached, a semantic search compares the current issue against past issues to count how many times the same problem has occurred
- If a premium customer has raised the same issue **2 or more times** across sessions, the backend overrides any other routing decision and **directly escalates to a human agent**

The memory is language-agnostic: issues are always stored in English (extracted by the RESOLVE block), so Hindi, Tamil, and Telugu conversations all contribute to the same searchable history.

### Demo-safe memory lifecycle
On **startup**: ChromaDB vector store and ticket history are wiped clean — every demo begins from a fresh state with no prior history.
On **shutdown**: the same wipe runs again, leaving no data behind.

### Speech Emotion Recognition (SER)
On every voice turn, audio is processed in parallel with STT — no added latency. The model `superb/wav2vec2-base-superb-er` classifies vocal tone into eight emotion labels:

| Emotion label | Frustration level |
|---|---|
| `ang` (angry) | critical |
| `dis` (disgust), `fea` (fearful) | high |
| `sad` | medium |
| `neu`, `hap`, `cal`, `sur` | low |

A **confidence gate** rejects weak detections: critical requires ≥ 65% confidence, high ≥ 55%, medium ≥ 50%. Anything below the gate falls back to `low`, preventing neutral speech from being mislabelled as angry.

The model is lazy-loaded on first voice call (~360 MB, cached afterwards). Set `USE_MOCK_SER=true` to skip it entirely.

### Dynamic ticket persistence
Tickets raised during a session are written to `data/mock/tickets.json` immediately. On the next session for the same customer, `get_tickets()` reads the file and the open ticket triggers the "ticket exists" branch — Aria acknowledges it without raising a duplicate.

### Indian language support
Speech-to-text uses Sarvam **saaras:v3**, purpose-built for Indian languages and Hinglish code-switching. Text-to-speech uses Sarvam **bulbul:v2**. Supported languages: Hindi, English, Tamil, Telugu, Marathi, Bengali.

### Structured handoff brief
On escalation, the human agent receives:
- Customer name, mobile, ARPU tier, plan, tenure
- Priority tier (P1 / P2 / P3) and score
- Detected issue and frustration level + signals
- Similar past issues from memory with dates
- `recurring_issue: true/false` flag
- Full conversation transcript (labelled Customer / Aria)
- Suggested first move ("CHURN RISK — do NOT use a script" or "Do NOT ask them to explain again")

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           TeleSupport AI                             │
│                                                                      │
│  APP STARTUP                                                         │
│  ├── Wipe ChromaDB vector store                                      │
│  └── Reset tickets.json  →  fresh demo slate                        │
│                                                                      │
│  /api/conversation/start                                             │
│         │                                                            │
│         ▼                                                            │
│  initialize()  ──── parallel fetch ──────────────────────────────   │
│         │           ├── CRM profile       (customers.json)          │
│         │           ├── Ticket history    (tickets.json)            │
│         │           └── Past issues       (ChromaDB)                │
│         │                                                            │
│  Gemini 2.5 Flash  ◄── system prompt:                               │
│  multi-turn chat        Aria persona + ARPU tier rules +            │
│                         frustration thresholds + memory context     │
│         │                                                            │
│  ══════════════════ CONVERSATION LOOP ═══════════════════           │
│  ║                                                                   │
│  ║  VOICE TURN                          TEXT TURN                   │
│  ║  ├── Sarvam STT ─┐ parallel          └── typed text             │
│  ║  └── SER (wav2vec2) ┘                                            │
│  ║                                                                   │
│  ║  _detect_frustration(text, audio_emotion)                        │
│  ║  ├── Text: keywords + expletives + insults + churn words         │
│  ║  ├── SER:  emotion label × confidence gate                       │
│  ║  └── Takes the higher of the two signals                         │
│  ║                │                                                  │
│  ║  [Frustration: low / medium / high / critical]                   │
│  ║                │                                                  │
│  ║  send_message_async()  ──►  Gemini 2.5 Flash                     │
│  ║                │                                                  │
│  ║  RESOLVE:{...} found?                                             │
│  ║  NO  ──► return spoken text, continue loop                       │
│  ║  YES ──► _close()                                                │
│  ║                                                                   │
│  ═══════════════════════════════════════════════════════            │
│                                                                      │
│  _close()                                                            │
│  ├── recall_similar_complaints()  ──►  ChromaDB semantic search     │
│  │                                                                   │
│  ├── Routing overrides (in priority order):                         │
│  │   1. Churn risk                  → escalate  (all tiers)         │
│  │   2. Premium + 2+ past issues    → escalate                      │
│  │   3. Premium + medium+ frustration → escalate                    │
│  │   4. Standard + high/critical    → escalate                      │
│  │   5. Standard + medium           → raise ticket + assurance      │
│  │                                                                   │
│  ├── raise_ticket ──► Write to tickets.json  (CRM mock)             │
│  │                ──► Send SMS  (mock)                              │
│  │                ──► store_conversation()  ──►  ChromaDB           │
│  │                ──► follow-up question  (loop continues)          │
│  │                                                                   │
│  ├── ticket_exists ──► Confirm open ticket ID                       │
│  │                 ──► follow-up question  (loop continues)         │
│  │                                                                   │
│  └── escalate_human ──► Build handoff brief                         │
│                     ──► store_conversation()  ──►  ChromaDB         │
│                     ──► End session                                  │
│                                                                      │
│  APP SHUTDOWN                                                        │
│  ├── Wipe ChromaDB vector store                                      │
│  └── Reset tickets.json                                              │
└──────────────────────────────────────────────────────────────────────┘
```

The legacy **LangGraph pipeline** (`agent/graph.py`) remains in the codebase as the formal thesis architecture diagram and powers the single-shot `/api/message/text` and `/api/message/voice` endpoints. The conversational endpoints (`/api/conversation/*`) use `agent/conversation.py` directly and bypass the graph.

---

## Project Structure

```
TelecomAgent/
├── main.py                        # FastAPI app + startup/shutdown memory lifecycle
├── requirements.txt
├── .env.example
│
├── agent/
│   ├── conversation.py            # Multi-turn conversational agent (primary)
│   ├── conditions.py              # Priority scoring, routing helpers
│   ├── graph.py                   # LangGraph pipeline (thesis diagram)
│   ├── state.py                   # SupportAgentState TypedDict
│   └── nodes/                     # LangGraph node implementations
│
├── api/
│   ├── routes.py                  # FastAPI route definitions
│   └── schemas.py                 # Request/response Pydantic models
│
├── services/
│   ├── crm.py                     # CRM mock — reads/writes customers.json + tickets.json
│   ├── sarvam.py                  # Sarvam STT (saaras:v3)
│   ├── tts.py                     # Sarvam TTS (bulbul:v2)
│   ├── ser.py                     # Speech Emotion Recognition (wav2vec2-base-superb-er)
│   ├── vector_db.py               # ChromaDB + all-MiniLM-L6-v2 cross-session memory
│   ├── sms.py                     # SMS mock
│   └── session_store.py           # In-memory session registry
│
├── data/
│   ├── mock/
│   │   ├── customers.json         # 4 demo customer profiles
│   │   └── tickets.json           # Live ticket store — written to at runtime
│   ├── chroma_db/                 # ChromaDB vector store — created at runtime
│   └── greetings/                 # Pre-cached greeting audio (hi/en/ta/te/mr/bn)
│
├── telecom_support_prototype.html # Demo chat UI (served at /)
└── tests/
    ├── test_ser.py                # SER pipeline verification
    └── test_graph.py
```

---

## Demo Customer Profiles

All customers start with zero tickets and zero memory on each server run.

| ID | Name | Plan | Tier | Demo scenario |
|---|---|---|---|---|
| CUST001 | Rahul Verma | Airtel Black ₹1499 | `black` | Premium — escalates at medium frustration; after 2 sessions with same issue, skips ticket and escalates directly |
| CUST002 | Priya Sharma | Prepaid ₹299 | `prepaid_monthly` | Standard — ticket raised with warm assurance; only escalates if highly agitated |
| CUST003 | Suresh Iyer | Postpaid ₹999 | `postpaid_high` | Premium long tenure — second session with same issue triggers direct escalation |
| CUST004 | Deepak Nair | Postpaid ₹549 | `postpaid_std` | Standard — escalates only on high/critical frustration or persistent agitation |

---

## Requirements

- Python 3.11 or higher
- A Gemini API key (Google AI Studio — free tier works)
- A Sarvam API key (sarvam.ai — required for voice STT and TTS)

### Python dependencies

```
fastapi >= 0.111
uvicorn >= 0.29
langgraph >= 0.1
langchain-google-genai >= 1.0
google-generativeai >= 0.5
httpx >= 0.27
python-multipart >= 0.0.9
pydantic >= 2.0
python-dotenv >= 1.0
sarvamai
transformers >= 4.40      # SER model (wav2vec2)
av >= 12.0                # WebM/Opus audio decoding — no system FFmpeg needed
chromadb >= 0.5           # Cross-session vector memory
sentence-transformers >= 3.0  # all-MiniLM-L6-v2 embeddings
```

> **SER model (~360 MB):** downloaded on first voice call, cached by HuggingFace. Set `USE_MOCK_SER=true` to skip.
> **Embedding model (~80 MB):** downloaded on first conversation with `USE_MOCK_VECTOR_DB=false`, cached by sentence-transformers.

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd TelecomAgent
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
GEMINI_API_KEY=your_gemini_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here

# Mock flags
USE_MOCK_CRM=true
USE_MOCK_STT=false
USE_MOCK_VECTOR_DB=false   # false = real ChromaDB memory; true = skip
USE_MOCK_SMS=true
USE_MOCK_SER=false         # false = real SER model; true = skip
```

> Set `USE_MOCK_STT=true` to run without a Sarvam key (typed input only).

### 3. (Optional) Verify SER is working

```bash
python tests/test_ser.py
```

Downloads the wav2vec2 model on first run (~360 MB), runs a synthetic signal through the full pipeline, and prints emotion scores. Skip if `USE_MOCK_SER=true`.

### 4. Run the server

```bash
python main.py
```

This starts uvicorn with hot-reload watching only source directories (`agent/`, `api/`, `services/`, `models/`). Writes to `data/` — tickets, ChromaDB — do not trigger a server restart and do not reset the loaded models.

### 5. Open the demo UI

Visit [http://localhost:8000](http://localhost:8000) in your browser.

Select a customer profile and language, press **Start Conversation**, then type or speak your complaint.

> Every time you restart the server, ChromaDB and tickets reset automatically — the demo always starts fresh.

---

## API Endpoints

### Conversational (multi-turn)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/conversation/start` | Start session, load CRM + memory, return greeting |
| `POST` | `/api/conversation/{session_id}/text` | Send a typed message |
| `POST` | `/api/conversation/{session_id}/voice` | Send a voice recording (WebM) — runs STT + SER in parallel |
| `DELETE` | `/api/conversation/{session_id}` | End and clean up session |
| `POST` | `/api/tts/stream` | Stream TTS audio for any text |

### Single-shot (LangGraph pipeline)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/message/text` | Full pipeline, single turn, text input |
| `POST` | `/api/message/voice` | Full pipeline, single turn, voice input |
| `POST` | `/api/session/start` | Return greeting audio URL |

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## How the RESOLVE Signal Works

Aria's responses are plain conversational text. When she is ready to take an action — raise a ticket, confirm an existing one, escalate, or close — she appends a structured JSON block to the same message:

```
RESOLVE:{"intent":"5G connectivity failure","frustration_level":"high",
"recommended_action":"raise_ticket","priority_score":7,
"churn_risk":false,"frustration_signals":["3 days without service","called twice before"]}
```

The backend strips this block from the spoken text before it reaches the customer, parses it, applies tier-based and memory-based routing overrides, executes the action (CRM write, SMS, ChromaDB store), and returns the result to the frontend.

`intent` is always in English regardless of the conversation language. This means cross-language semantic recall works correctly: a Hindi session and a Tamil session about the same 5G problem will both match "5G connectivity failure" in ChromaDB.

---

## Supported Languages

| Code | Language | STT | TTS | Frustration keywords |
|---|---|---|---|---|
| `en` | English | ✅ | ✅ | ✅ |
| `hi` | Hindi | ✅ | ✅ | ✅ (Hindi insults + Hinglish) |
| `ta` | Tamil | ✅ | ✅ | ✅ (via SER voice signal) |
| `te` | Telugu | ✅ | ✅ | ✅ (via SER voice signal) |
| `mr` | Marathi | ✅ | ✅ | ✅ (via SER voice signal) |
| `bn` | Bengali | ✅ | ✅ | ✅ (via SER voice signal) |

---

## Thesis Context

This prototype was built to evaluate whether a conversational AI agent can meaningfully reduce support handling time and improve routing accuracy in an Indian telecom context.

**Research questions addressed:**
1. Can Gemini 2.5 Flash reliably classify intent and frustration from Hinglish/multilingual input?
2. Does ARPU-tier-aware routing reduce unnecessary escalations for low-value customers while preserving response quality for premium ones?
3. Does cross-session memory (ChromaDB) improve routing accuracy for recurring issues — specifically, does detecting the same issue across sessions lead to faster escalation for premium customers?
4. Does a structured handoff brief reduce the time a human agent needs to understand the customer's context?

**Evaluation dataset:** `data/airtel_support_cluster_reviews.xlsx` — 288 manually labelled Airtel app reviews with expected routing actions (`raise_ticket` / `update_existing` / `escalate_human`). Run these through the `/api/message/text` endpoint and compare `resolution_branch` against the labels for F1 score.
