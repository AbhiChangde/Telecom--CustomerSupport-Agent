# TeleSupport AI

An intelligent, multi-turn voice and text customer support agent for Indian telecom operators, modelled on Airtel. Built as a thesis prototype demonstrating how conversational AI can eliminate friction in telecom support — from re-identification and language barriers to unstructured escalations and missed context.

---

## The Problem

Customer support in Indian telecom is broken in predictable ways:

- **Re-identification on every call.** Customers repeat their name, number, and issue to each agent they reach, even when they have called before.
- **Language mismatch.** IVR systems and agents often cannot handle Hinglish, Tamil, or Telugu code-switching that customers naturally use.
- **Frustrated customers get generic responses.** High-value customers who are already agitated receive the same scripted reply as a first-time caller.
- **No structured handoff.** When a human agent finally takes the call, they receive no context — the customer has to start from scratch.
- **Ticket inflation.** Duplicate tickets are raised for the same ongoing issue because agents cannot see history in real time.

TeleSupport AI addresses each of these by loading customer context before the first word is spoken, adapting its tone and routing based on the customer's frustration level and ARPU tier, and handing off a detailed brief to the human agent if escalation is needed.

---

## What It Does

The agent is named **Aria**. She conducts a short, empathetic spoken or typed conversation with the customer, then routes the interaction to one of three outcomes:

| Outcome | When |
|---|---|
| **Raise ticket** | New issue not already tracked |
| **Confirm existing ticket** | Open ticket already exists for this issue |
| **Escalate to human** | Premium customer with frustration, or any customer who is highly agitated / at churn risk |

The human agent receives a structured **handoff brief** — customer profile, ARPU tier, conversation summary, detected frustration signals, prior unresolved ticket count, and a suggested first move — so they never have to ask the customer to repeat themselves.

---

## Key Features

### Conversational, multi-turn flow
Unlike a single-shot intent classifier, Aria holds a real multi-turn conversation via a persistent Gemini chat session. She greets, empathises, clarifies if needed, resolves, follows up ("Is there anything else I can help with?"), and only closes when the customer is satisfied or escalation is triggered.

### Per-turn frustration detection (dual-signal)
Every customer message is evaluated by two independent signals before it reaches Gemini:

| Signal | Source | How it works |
|---|---|---|
| **Text keywords** | Transcript or typed text | Scans for frustration words (`useless`, `fraud`, `port`, churn signals, etc.) — also works in Hindi (`बेकार`, `घटिया`) |
| **Voice emotion (SER)** | Raw audio only | `superb/wav2vec2-base-superb-er` classifies the audio into emotion categories (angry, sad, fearful, etc.) and maps them to frustration levels |

The higher of the two signals wins. A calm sentence spoken in an angry tone, or an aggressive sentence spoken quietly, will each be caught by the appropriate signal.

Each message is tagged `[Frustration: low/medium/high/critical]` before being sent to Gemini, so the LLM adapts its tone and routing decision on every single turn — not just at resolution time. A consecutive frustrated turn counter also tracks sustained agitation across turns.

### ARPU-tier-aware routing
Escalation thresholds differ by customer value:

| Tier | Escalation trigger |
|---|---|
| `black` / `postpaid_high` | Medium frustration or above |
| `postpaid_std` / `prepaid_*` | High or critical frustration only |

Lower-tier customers who are only mildly frustrated receive warm ticket assurance instead of being escalated. Even if Gemini recommends escalation for a low-ARPU customer with medium frustration, the backend overrides it to raise a ticket.

### Speech Emotion Recognition (SER)
On every voice turn, the raw audio is processed in parallel with STT — it never waits for transcription to finish. The model used is `superb/wav2vec2-base-superb-er`, a wav2vec2 model fine-tuned on the IEMOCAP emotion corpus. It classifies audio into eight emotion categories which are mapped to frustration levels:

| Emotion label | Frustration level |
|---|---|
| `ang` (angry) | critical |
| `dis` (disgust), `fea` (fearful) | high |
| `sad` | medium |
| `neu`, `hap`, `cal`, `sur` | low |

A **confidence gate** filters out weak detections: critical requires ≥ 65% confidence, high requires ≥ 55%, medium requires ≥ 50%. Anything below the gate falls back to `low`. This prevents mislabelling neutral speech that happens to score a low-confidence `ang`.

The model is lazy-loaded on first voice call (≈ 360 MB download, cached afterwards). Set `USE_MOCK_SER=true` in `.env` to skip the model entirely.

SER runs on CPU with no GPU required. Every voice turn prints a terminal log showing all four top emotion scores and the final frustration decision.

### Indian language support
Speech-to-text uses Sarvam **saaras:v3**, purpose-built for Indian languages and Hinglish code-switching. Text-to-speech uses Sarvam **bulbul:v2**. Supported languages: Hindi, English, Tamil, Telugu, Marathi, Bengali.

### Structured handoff brief
On escalation, the human agent receives:
- Customer name, mobile, ARPU tier, plan, tenure
- Priority tier (P1 / P2 / P3) and score
- Detected issue and frustration level
- Full conversation transcript (labelled Customer / Aria)
- Suggested first move ("Do NOT ask them to explain again")

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Conversational Agent                      │
│                                                             │
│  /conversation/start                                        │
│       │                                                     │
│       ▼                                                     │
│  initialize()  ──── parallel ──── CRM profile              │
│       │                      └─── Ticket history           │
│       │                                                     │
│       ▼                                                     │
│  Gemini 2.5 Flash  ◄──── system prompt (Aria persona,      │
│  multi-turn chat         ARPU tier rules, tone detection)  │
│       │                                                     │
│  Each voice turn:                                           │
│  STT (Sarvam) ──── parallel ────  SER (wav2vec2)           │
│       │                                │                    │
│  transcript                    frustration level            │
│       │                                │                    │
│       └──────────── combined ──────────┘                    │
│  Each turn (text or voice):                                 │
│  1. _detect_frustration(text, audio_emotion)                │
│     → higher of keyword scan + SER signal wins             │
│     → tags message [Frustration: X]                        │
│  2. send_message_async() → Gemini responds                  │
│  3. RESOLVE:{...} detected? → _close()                     │
│       │                                                     │
│       ├─── raise_ticket  → CRM + SMS + follow-up           │
│       ├─── ticket_exists → confirm + follow-up             │
│       ├─── escalate_human → handoff brief + end            │
│       └─── close         → graceful end                    │
└─────────────────────────────────────────────────────────────┘
```

The legacy **LangGraph pipeline** (`agent/graph.py`) remains in the codebase as the formal thesis architecture diagram and powers the single-shot `/api/message/text` and `/api/message/voice` endpoints. The conversational endpoints (`/api/conversation/*`) use `agent/conversation.py` directly and bypass the graph.

---

## Project Structure

```
TelecomAgent/
├── main.py                        # FastAPI app entrypoint
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
│   ├── crm.py                     # CRM mock (reads data/mock/customers.json)
│   ├── sarvam.py                  # Sarvam STT (saaras:v3)
│   ├── tts.py                     # Sarvam TTS (bulbul:v2)
│   ├── ser.py                     # Speech Emotion Recognition (wav2vec2-base-superb-er)
│   ├── vector_db.py               # Pinecone / mock vector memory
│   ├── sms.py                     # SMS mock
│   └── session_store.py           # In-memory session registry
│
├── data/
│   ├── mock/
│   │   ├── customers.json         # 4 demo customer profiles
│   │   └── tickets.json           # Sample ticket history
│   └── greetings/                 # Pre-cached greeting audio (hi/en/ta/te/mr/bn)
│
├── telecom_support_prototype.html # Demo chat UI (served at /)
└── tests/
```

---

## Demo Customer Profiles

| ID | Name | Plan | Tier | Scenario |
|---|---|---|---|---|
| CUST001 | Rahul Verma | Airtel Black ₹1499 | `black` | Premium + prior unresolved → escalates at medium frustration |
| CUST002 | Priya Sharma | Prepaid ₹299 | `prepaid_monthly` | New customer → ticket raised with assurance |
| CUST003 | Suresh Iyer | Postpaid ₹999 | `postpaid_high` | Long tenure + open ticket → existing ticket confirmed |
| CUST004 | Deepak Nair | Postpaid ₹549 | `postpaid_std` | 3 unresolved tickets → escalates if frustration is high |

---

## Requirements

- Python 3.11 or higher
- A Gemini API key (Google AI Studio — free tier works)
- A Sarvam API key (sarvam.ai — required for voice STT and TTS)
- Pinecone API key (optional — set `USE_MOCK_VECTOR_DB=true` to skip)

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
pinecone-client >= 3.0
python-dotenv >= 1.0
sarvamai
transformers >= 4.40   # SER model
av >= 12.0             # WebM/Opus audio decoding (no system FFmpeg needed)
```

> **Note on SER model size:** `transformers` downloads `superb/wav2vec2-base-superb-er` (~360 MB) on first voice call. It is cached afterwards by HuggingFace. If you want to skip this entirely, set `USE_MOCK_SER=true` — no `transformers` or `torch` installation needed in that case.

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd TelecomAgent
python -m venv venv
```

Activate the environment:

```bash
# Windows (PowerShell)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1

# Windows (Command Prompt)
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
GEMINI_API_KEY=your_gemini_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here

# Pinecone (optional — set USE_MOCK_VECTOR_DB=true to skip)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=telesupport-memory
VECTOR_DB_PROVIDER=pinecone

# Mock flags — set true to run without external services
USE_MOCK_CRM=true
USE_MOCK_STT=false
USE_MOCK_VECTOR_DB=true
USE_MOCK_SMS=true
USE_MOCK_SER=false     # set true to skip SER model (no torch needed)
```

> To run the demo without any API keys at all, set `USE_MOCK_STT=true` in addition to the others. Set `USE_MOCK_SER=true` if you want to skip the 360 MB model download.

### 4. (Optional) Verify SER is working

```bash
python tests/test_ser.py
```

This downloads the wav2vec2 model on first run (~360 MB), runs a synthetic audio signal through the full pipeline, and prints the emotion scores. Skip this step if you set `USE_MOCK_SER=true`.

### 5. Run the server

```bash
uvicorn main:app --reload
```

### 6. Open the demo UI

Visit [http://localhost:8000](http://localhost:8000) in your browser.

Select a customer profile and language, press **Start Conversation**, then type or speak your complaint.

---

## API Endpoints

### Conversational (multi-turn)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/conversation/start` | Start a session, load customer context, return greeting |
| `POST` | `/api/conversation/{session_id}/text` | Send a typed message |
| `POST` | `/api/conversation/{session_id}/voice` | Send a voice recording (WebM) |
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

Aria's responses are plain conversational text. When she is ready to take an action (raise a ticket, confirm an existing one, escalate, or close), she appends a structured JSON block to her message:

```
RESOLVE:{"intent":"5G connectivity failure","frustration_level":"high",
"recommended_action":"raise_ticket","priority_score":7,
"churn_risk":false,"frustration_signals":["3 days without service","called twice before"]}
```

The backend strips this block from the spoken text, parses it, applies tier-based routing overrides, executes the action (CRM ticket, SMS, vector store), and returns the result to the frontend. The customer never sees the RESOLVE block.

---

## Supported Languages

| Code | Language | STT | TTS |
|---|---|---|---|
| `en` | English | ✅ | ✅ |
| `hi` | Hindi | ✅ | ✅ |
| `ta` | Tamil | ✅ | ✅ |
| `te` | Telugu | ✅ | ✅ |
| `mr` | Marathi | ✅ | ✅ |
| `bn` | Bengali | ✅ | ✅ |

---

## Thesis Context

This prototype was built to evaluate whether a conversational AI agent can meaningfully reduce support handling time and improve routing accuracy in an Indian telecom context.

**Research questions addressed:**
1. Can Gemini Flash reliably classify intent and frustration from Hinglish/multilingual input?
2. Does ARPU-tier-aware routing reduce unnecessary escalations for low-value customers while preserving response quality for premium ones?
3. Does a structured handoff brief reduce the time a human agent needs to understand the customer's context?

**Evaluation dataset:** `data/airtel_support_cluster_reviews.xlsx` — 288 manually labelled Airtel app reviews with expected routing actions (`raise_ticket` / `update_existing` / `escalate_human`). Run these through the `/api/message/text` endpoint and compare `resolution_branch` against the labels for F1 score.
