"""
Run this script directly to verify the SER pipeline works end-to-end.

    python tests/test_ser.py

It will:
  1. Check imports (av, transformers, torch)
  2. Decode a synthetic WebM-like audio signal
  3. Load the wav2vec2 emotion model (downloads ~360 MB on first run)
  4. Run inference and print the top-5 emotion probabilities
  5. Show the final frustration_level that would be sent to Gemini
"""

import sys
import struct
import numpy as np

PASS = "  [PASS]"
FAIL = "  [FAIL]"
LINE = "-" * 55


def section(title: str):
    print(f"\n{LINE}\n  {title}\n{LINE}")


# ── 1. Import checks ──────────────────────────────────────────────────────────
section("1. Import checks")

try:
    import av
    print(f"{PASS} av (PyAV) {av.__version__}")
except ImportError as e:
    print(f"{FAIL} av -- {e}")
    print("       Run: pip install av")
    sys.exit(1)

try:
    import torch
    print(f"{PASS} torch {torch.__version__}  (CUDA: {torch.cuda.is_available()})")
except ImportError as e:
    print(f"{FAIL} torch -- {e}")
    print("       Run: pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

try:
    import transformers
    print(f"{PASS} transformers {transformers.__version__}")
except ImportError as e:
    print(f"{FAIL} transformers -- {e}")
    print("       Run: pip install transformers")
    sys.exit(1)


# ── 2. Build a synthetic audio signal ────────────────────────────────────────
section("2. Synthetic audio (3 s sine wave @ 16 kHz)")

SAMPLE_RATE = 16_000
DURATION    = 3  # seconds

t = np.linspace(0, DURATION, SAMPLE_RATE * DURATION, endpoint=False)
# Mix of two tones to simulate speech-like energy
audio_np = (0.4 * np.sin(2 * np.pi * 220 * t) +
            0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
print(f"{PASS} Generated {len(audio_np):,} samples  shape={audio_np.shape}  dtype={audio_np.dtype}")


# ── 3. Encode to WAV bytes (simulates what the browser sends, roughly) ────────
section("3. Encode to WAV bytes via av")

import io

def ndarray_to_wav_bytes(array: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="wav")
    stream = container.add_stream("pcm_s16le", rate=sample_rate, layout="mono")
    frame = av.AudioFrame.from_ndarray(
        (array * 32767).astype(np.int16).reshape(1, -1),
        format="s16",
        layout="mono",
    )
    frame.sample_rate = sample_rate
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return buf.getvalue()

wav_bytes = ndarray_to_wav_bytes(audio_np, SAMPLE_RATE)
print(f"{PASS} Encoded to WAV bytes -- {len(wav_bytes):,} bytes")


# ── 4. Decode back with _decode_webm (the same function used in production) ───
section("4. Decode bytes to numpy (production code path)")

sys.path.insert(0, ".")           # make project root importable
from services.ser import _decode_webm

decoded = _decode_webm(wav_bytes)
print(f"{PASS} Decoded: {len(decoded):,} samples  shape={decoded.shape}  dtype={decoded.dtype}")
print(f"       min={decoded.min():.4f}  max={decoded.max():.4f}  rms={np.sqrt(np.mean(decoded**2)):.4f}")


# ── 5. Load the model ─────────────────────────────────────────────────────────
section("5. Load wav2vec2-base-superb-er  (downloads ~360 MB on first run)")

print("  Loading... (this may take a minute the first time)")
from services.ser import _load_classifier
clf = _load_classifier()
print(f"{PASS} Model loaded: {clf.model.__class__.__name__}")


# ── 6. Run inference ──────────────────────────────────────────────────────────
section("6. Run inference")

results = clf({"array": decoded, "sampling_rate": SAMPLE_RATE}, top_k=8)
print(f"\n  Top emotions detected:\n")
for r in results:
    bar = "#" * int(r["score"] * 30)
    print(f"  {r['label']:<12} {r['score']:.3f}  {bar}")

top = results[0]
print(f"\n  Winning label : {top['label']}  (confidence {top['score']:.3f})")


# ── 7. Run the full classify_emotion() function ───────────────────────────────
section("7. Full classify_emotion() pipeline")

import asyncio
from services.ser import classify_emotion

result = asyncio.run(classify_emotion(wav_bytes))
print(f"\n  emotion           : {result['emotion']}")
print(f"  confidence        : {result['confidence']}")
print(f"  frustration_level : {result['frustration_level']}")
print(f"\n  This is the value that gets tagged onto each user voice message")
print(f"  before being sent to Gemini.")


# ── Summary ───────────────────────────────────────────────────────────────────
section("Summary")
print("  All checks passed. SER is working correctly.\n")
print("  Note: a synthetic sine wave will typically score as 'neutral' or 'calm'.")
print("  Real speech (especially angry/frustrated speech) will give more")
print("  meaningful emotion labels.\n")
