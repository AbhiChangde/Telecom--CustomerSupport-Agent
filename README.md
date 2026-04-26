# TeleSupport AI

Intelligent customer support agent for Indian telecom operators (modelled on Airtel).
Thesis prototype — a 6-state LangGraph pipeline with Sarvam STT, Gemini Flash, and async parallel context fetching.

## Quick start (all mocks, no API keys needed)

```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API.

## Test a text message

```bash
curl -X POST "http://localhost:8000/api/message/text?customer_id=CUST001&language_code=en&text=My%205G%20is%20not%20working%20for%203%20days"
```

## Demo customer IDs

| ID | Name | Tier | Scenario |
|---|---|---|---|
| CUST001 | Rahul Verma | Black (₹1499) | High ARPU + prior unresolved tickets → likely escalation |
| CUST002 | Priya Sharma | Prepaid Monthly (₹299) | New customer, no ticket history → raise ticket |
| CUST003 | Suresh Iyer | Postpaid High (₹999) | Long tenure + open ticket → ticket_exists branch |
| CUST004 | Deepak Nair | Postpaid Std (₹549) | Multiple unresolved tickets → frustration override |

## Architecture

```
language → greeting → input → context → decision
                                              ↓
                          raise_ticket / ticket_exists / escalate
```

State 4 (`context`) fires three parallel async calls (CRM profile, ticket history, vector memory).
State 5 (`decision`) makes one Gemini Flash call returning structured JSON.
State 6 routing uses a frustration override that can bypass Gemini's recommendation.

## Environment variables

See `.env.example`. Set all `USE_MOCK_*=true` for demo mode without API keys.
