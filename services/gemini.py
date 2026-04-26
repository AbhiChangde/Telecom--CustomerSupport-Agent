import os
import google.generativeai as genai

_model = None


def _get_model():
    global _model
    if _model is None:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        _model = genai.GenerativeModel("gemini-2.5-flash")
    return _model


async def call_gemini(prompt: str) -> str:
    """Single Gemini Flash call. Returns raw text response."""
    model = _get_model()
    response = await model.generate_content_async(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=800,
        ),
    )
    return response.text.strip()
