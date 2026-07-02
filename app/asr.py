import httpx
from app.config import SARVAM_API_KEY, SARVAM_STT_MODEL, SARVAM_STT_MODE

SARVAM_ASR_URL = "https://api.sarvam.ai/speech-to-text"

async def transcribe_audio(file_path: str) -> str:
    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        with open(file_path, "rb") as audio_file:
            files = {
                "file": ("audio.wav", audio_file, "audio/wav")
            }
            # Model/mode from config (.env) — migrate Sarvam STT versions without
            # code changes. `mode` is only valid on saaras:* models.
            data = {
                "model": SARVAM_STT_MODEL,
                # "unknown" → Sarvam auto-detects the language, matching the
                # production /ws pipeline. Hardcoding "en-IN" forced English
                # decoding and garbled Hindi/Marathi callers (Bug #19).
                "language_code": "unknown",
            }
            if SARVAM_STT_MODEL.startswith("saaras"):
                data["mode"] = SARVAM_STT_MODE

            response = await client.post(
                SARVAM_ASR_URL,
                headers=headers,
                files=files,
                data=data,
            )

    print("ASR status:", response.status_code)
    print("ASR response:", response.text)
    response.raise_for_status()
    result = response.json()

    return result.get("transcript", "")