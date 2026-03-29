import httpx
import base64
from app.config import SARVAM_API_KEY

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

async def text_to_speech(text: str, output_file: str = "output.wav") -> str:
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }

    # Sarvam TTS expects: inputs (list), target_language_code, speaker, model
    payload = {
        "inputs": [text],
        "target_language_code": "en-IN",
        "speaker": "anushka",
        "model": "bulbul:v2"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            SARVAM_TTS_URL,
            headers=headers,
            json=payload
        )

    print("TTS status:", response.status_code)
    print("TTS response preview:", response.text[:200])
    response.raise_for_status()

    # Sarvam returns JSON: {"audios": ["<base64-encoded-wav>"]}
    result = response.json()
    audio_b64 = result["audios"][0]
    audio_bytes = base64.b64decode(audio_b64)

    with open(output_file, "wb") as f:
        f.write(audio_bytes)

    return output_file