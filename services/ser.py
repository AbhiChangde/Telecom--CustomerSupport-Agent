import io
import os
import logging
import asyncio
import numpy as np
from functools import lru_cache

logger = logging.getLogger(__name__)

# Maps wav2vec2-SUPERB emotion labels → our frustration levels
_EMOTION_MAP = {
    "ang": "critical",   # angry
    "hap": "low",        # happy
    "neu": "low",        # neutral
    "sad": "medium",     # sad often signals helplessness / resignation
    "dis": "high",       # disgust
    "fea": "high",       # fearful (anxious)
    "cal": "low",        # calm
    "sur": "low",        # surprised
}

# Full label name fallback
_LABEL_MAP = {
    "angry":     "critical",
    "disgust":   "high",
    "fearful":   "high",
    "sad":       "medium",
    "neutral":   "low",
    "calm":      "low",
    "happy":     "low",
    "surprised": "low",
}


@lru_cache(maxsize=1)
def _load_classifier():
    """Lazy-loads the SER model once on first voice call. ~360 MB download."""
    from transformers import pipeline
    logger.info("Loading SER model (superb/wav2vec2-base-superb-er) — first call only…")
    clf = pipeline(
        "audio-classification",
        model="superb/wav2vec2-base-superb-er",
        device=-1,          # CPU inference
    )
    logger.info("SER model loaded.")
    return clf


def _decode_webm(audio_bytes: bytes) -> np.ndarray:
    """Decode WebM/Opus bytes → 16 kHz mono float32 numpy array using PyAV."""
    import av
    container = av.open(io.BytesIO(audio_bytes))
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
    samples: list[float] = []
    for frame in container.decode(audio=0):
        for resampled in resampler.resample(frame):
            samples.extend(resampled.to_ndarray().flatten().tolist())
    # Flush resampler
    for resampled in resampler.resample(None):
        samples.extend(resampled.to_ndarray().flatten().tolist())
    return np.array(samples, dtype=np.float32)


def _label_to_frustration(label: str, confidence: float) -> str:
    label = label.lower().strip()
    level = _EMOTION_MAP.get(label) or _LABEL_MAP.get(label, "low")
    # SER must clear a minimum confidence bar per level — weak signals → "low"
    # (neutral speech often gets a low-confidence "angry" score; ignore it)
    gates = {"critical": 0.65, "high": 0.55, "medium": 0.50}
    if level in gates and confidence < gates[level]:
        return "low"
    return level


async def classify_emotion(audio_bytes: bytes) -> dict:
    """
    Run SER on raw audio bytes.
    Returns: {emotion, frustration_level, confidence}
    Falls back to {emotion: "unknown", frustration_level: "low"} on any error.
    """
    if os.getenv("USE_MOCK_SER", "false").lower() == "true":
        return {"emotion": "neutral", "frustration_level": "low", "confidence": 1.0}

    try:
        # Audio decoding and model inference run in a thread pool
        # so they don't block the async event loop
        loop = asyncio.get_event_loop()
        audio_array = await loop.run_in_executor(None, _decode_webm, audio_bytes)

        def _infer():
            clf = _load_classifier()
            return clf({"array": audio_array, "sampling_rate": 16000}, top_k=4)

        results = await loop.run_in_executor(None, _infer)
        top        = results[0]
        emotion    = top["label"]
        confidence = round(float(top["score"]), 3)
        frustration = _label_to_frustration(emotion, confidence)

        # ── SER terminal log (always visible) ────────────────────────────
        bar = lambda s: "#" * int(s * 20)
        print("\n" + "=" * 48, flush=True)
        print("  SER RESULT", flush=True)
        print("=" * 48, flush=True)
        for r in results:
            marker = " <-- TOP" if r == top else ""
            print(f"  {r['label']:<6}  {r['score']:.3f}  {bar(r['score'])}{marker}", flush=True)
        print(f"  {'---'}", flush=True)
        print(f"  frustration_level : {frustration}", flush=True)
        print(f"  confidence gate   : {'PASSED' if frustration != 'low' or emotion in ('neu','hap','cal') else 'BLOCKED (too weak)'}", flush=True)
        print("=" * 48 + "\n", flush=True)
        # ─────────────────────────────────────────────────────────────────

        return {"emotion": emotion, "frustration_level": frustration, "confidence": confidence}

    except Exception as exc:
        print(f"\n[SER ERROR] {type(exc).__name__}: {exc}\n", flush=True)
        return {"emotion": "unknown", "frustration_level": "low", "confidence": 0.0}
