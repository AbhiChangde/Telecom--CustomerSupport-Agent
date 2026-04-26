import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from agent.graph import support_graph
from agent.state import SupportAgentState
from api.schemas import StartSessionResponse, AgentResponse
from services.sarvam import transcribe
from services import tts
from services import session_store

router = APIRouter()

LANGUAGE_NAMES = {
    "hi": "Hindi",
    "en": "English",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "bn": "Bengali",
}


@router.post("/session/start", response_model=StartSessionResponse)
async def start_session(customer_id: str, language_code: str):
    """State 1 + 2: Initialise session and return greeting audio URL."""
    if language_code not in LANGUAGE_NAMES:
        raise HTTPException(status_code=400, detail=f"Unsupported language_code: {language_code}")

    greeting_url = f"/static/greetings/{language_code}.mp3"
    return StartSessionResponse(
        session_id=str(uuid.uuid4()),
        customer_id=customer_id,
        language_code=language_code,
        greeting_url=greeting_url,
    )


@router.post("/message/voice", response_model=AgentResponse)
async def process_voice(
    customer_id: str = Form(...),
    language_code: str = Form(...),
    audio: UploadFile = File(...),
):
    """State 3 (voice path): Transcribe audio then run full pipeline."""
    if language_code not in LANGUAGE_NAMES:
        raise HTTPException(status_code=400, detail=f"Unsupported language_code: {language_code}")

    audio_bytes = await audio.read()
    transcript = await transcribe(audio_bytes, language_code)

    initial_state = _build_initial_state(customer_id, language_code, transcript, "voice")
    result = await support_graph.ainvoke(initial_state)

    return _format_response(result, transcript)


@router.post("/message/text", response_model=AgentResponse)
async def process_text(customer_id: str, language_code: str, text: str):
    """State 3 (text path): Run full pipeline with typed input."""
    if language_code not in LANGUAGE_NAMES:
        raise HTTPException(status_code=400, detail=f"Unsupported language_code: {language_code}")

    initial_state = _build_initial_state(customer_id, language_code, text, "text")
    result = await support_graph.ainvoke(initial_state)

    return _format_response(result)


def _build_initial_state(customer_id, language_code, user_input, input_mode) -> SupportAgentState:
    return SupportAgentState(
        customer_id=customer_id,
        language_code=language_code,
        language_name=LANGUAGE_NAMES.get(language_code, "English"),
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


def _format_response(result: SupportAgentState, transcript: str = None) -> AgentResponse:
    decision = result.get("decision")
    priority_score = decision["priority_score"] if decision else None

    def _get_tier(score: int) -> str:
        if score >= 8:
            return "P1"
        if score >= 5:
            return "P2"
        return "P3"

    return AgentResponse(
        response=result["response_text"],
        resolution_branch=result.get("resolution_branch"),
        priority_score=priority_score,
        priority_tier=_get_tier(priority_score) if priority_score else None,
        intent=decision["intent"] if decision else None,
        transcript=transcript,
        ticket_id=result.get("new_ticket_id"),
        handoff_brief=result.get("handoff_brief"),
    )


# ── Conversational endpoints ─────────────────────────────────────────────────

@router.post("/conversation/start")
async def conversation_start(customer_id: str, language_code: str):
    """Create a session, load customer context, return greeting text + audio."""
    if language_code not in LANGUAGE_NAMES:
        raise HTTPException(status_code=400, detail=f"Unsupported language_code: {language_code}")

    session_id = str(uuid.uuid4())
    agent = session_store.create(
        session_id, customer_id, language_code, LANGUAGE_NAMES[language_code]
    )
    greeting_text = await agent.initialize()

    return {
        "session_id":   session_id,
        "customer_id":  customer_id,
        "language_code": language_code,
        "greeting":     greeting_text,
    }


@router.post("/conversation/{session_id}/voice")
async def conversation_voice(
    session_id: str,
    language_code: str = Form(...),
    audio: UploadFile = File(...),
):
    """Voice turn: transcribe audio then send to conversation agent."""
    agent = session_store.get(session_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    if agent.is_resolved:
        raise HTTPException(status_code=400, detail="Conversation already resolved")

    audio_bytes = await audio.read()
    transcript  = await transcribe(audio_bytes, language_code)
    result      = await agent.send_message(transcript)
    result["transcript"] = transcript
    return result


@router.post("/conversation/{session_id}/text")
async def conversation_text(session_id: str, text: str):
    """Text turn: send typed message to conversation agent."""
    agent = session_store.get(session_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    if agent.is_resolved:
        raise HTTPException(status_code=400, detail="Conversation already resolved")

    result = await agent.send_message(text)
    return result


@router.delete("/conversation/{session_id}")
async def conversation_end(session_id: str):
    """Clean up session after conversation ends."""
    session_store.delete(session_id)
    return {"status": "deleted"}


@router.post("/tts/stream")
async def tts_stream(text: str, language_code: str = "en"):
    """Stream TTS audio from Sarvam bulbul:v3 for the agent's response text."""
    if language_code not in LANGUAGE_NAMES:
        language_code = "en"

    return StreamingResponse(
        tts.text_to_speech_stream(text, language_code),
        media_type="audio/mpeg",
    )
