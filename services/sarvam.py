import os
import io
from sarvamai import SarvamAI

MOCK_TRANSCRIPTS = {
    "hi": "Mera 5G chal nahi raha pichle 3 din se. Data bhi bahut jaldi khatam ho raha hai.",
    "en": "My 5G is showing but data is still getting used up. This has been happening for 3 days.",
    "ta": "என் 5G சேவை 3 நாட்களாக செயல்படவில்லை.",
    "te": "నా 5G 3 రోజులుగా పని చేయడం లేదు.",
    "mr": "माझे 5G 3 दिवसांपासून काम करत नाही.",
    "bn": "আমার 5G 3 দিন ধরে কাজ করছে না।",
}


async def transcribe(audio_bytes: bytes, language_code: str) -> str:
    """
    Transcribe audio using Sarvam saaras:v3.
    language_code: hi, en, ta, te, mr, bn
    """
    if os.getenv("USE_MOCK_STT", "true").lower() == "true":
        return MOCK_TRANSCRIPTS.get(language_code, MOCK_TRANSCRIPTS["en"])

    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise ValueError("SARVAM_API_KEY is not set in .env")

    client = SarvamAI(api_subscription_key=api_key)

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "recording.webm"

    response = client.speech_to_text.transcribe(
        file=audio_file,
        model="saaras:v3",
        mode="transcribe",
    )

    return response.transcript
