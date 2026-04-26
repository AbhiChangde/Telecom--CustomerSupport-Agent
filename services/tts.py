import base64
import os
import logging
import httpx

logger = logging.getLogger(__name__)

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

# All valid speakers confirmed from Sarvam API error response.
# Using language-appropriate female speakers for Aria.
# anushka  → general female (hi, en, mr, bn)
# kavitha  → South Indian female (te, ta)
LANGUAGE_SPEAKER_MAP = {
    "hi": ("hi-IN", "anushka"),
    "en": ("en-IN", "anushka"),
    "ta": ("ta-IN", "anushka"),
    "te": ("te-IN", "anushka"),
    "mr": ("mr-IN", "anushka"),
    "bn": ("bn-IN", "anushka"),
}


async def text_to_speech_stream(text: str, language_code: str):
    """Calls Sarvam REST TTS and yields MP3 audio bytes."""
    if os.getenv("USE_MOCK_TTS", "false").lower() == "true":
        yield b""
        return

    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise ValueError("SARVAM_API_KEY is not set in .env")

    target_lang, speaker = LANGUAGE_SPEAKER_MAP.get(language_code, ("en-IN", "anushka"))
    logger.info(f"TTS → lang={target_lang} speaker={speaker} text_len={len(text)}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        for chunk in _split_text(text, max_chars=500):
            response = await client.post(
                SARVAM_TTS_URL,
                headers={
                    "api-subscription-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": [chunk],
                    "target_language_code": target_lang,
                    "speaker": speaker,
                    "model": "bulbul:v2",
                },
            )
            if not response.is_success:
                print("=" * 60, flush=True)
                print(f"TTS FAILED: lang={target_lang} speaker={speaker}", flush=True)
                print(f"STATUS: {response.status_code}", flush=True)
                print(f"BODY: {response.text}", flush=True)
                print("=" * 60, flush=True)
                response.raise_for_status()
            for audio_b64 in response.json().get("audios", []):
                yield base64.b64decode(audio_b64)


def _split_text(text: str, max_chars: int = 500) -> list[str]:
    """Split at sentence boundaries to stay within API limits."""
    if len(text) <= max_chars:
        return [text]
    parts, current = [], ""
    for sentence in text.replace("। ", ". ").split(". "):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 2 <= max_chars:
            current += ("" if not current else ". ") + sentence
        else:
            if current:
                parts.append(current)
            current = sentence
    if current:
        parts.append(current)
    return parts or [text[:max_chars]]
