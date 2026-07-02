from openai import AsyncOpenAI
from loguru import logger
from app.config import GEMINI_API_KEY, GEMINI_BASE_URL, VOICE_LLM_MODEL

# Legacy one-shot helper for the POST /voice REST endpoint (NOT the production
# /ws streaming path). Uses the same voice LLM as the live path — Gemini via its
# OpenAI-compatible endpoint. A 15s timeout keeps a stuck request from hanging
# the handler far longer than a phone caller would wait.
client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL, timeout=15.0)

# reasoning_effort is a gpt-oss (Groq) param; Gemini's OpenAI-compatible endpoint
# rejects unknown params, so include it only when the voice model is gpt-oss.
_VOICE_REASONING_KW = (
    {"reasoning_effort": "low"}
    if ("gpt-oss" in VOICE_LLM_MODEL.lower() or VOICE_LLM_MODEL.lower().startswith("openai/"))
    else {}
)

# Spoken when the model returns nothing or errors, so the endpoint never crashes
# or returns dead audio.
_FALLBACK = "Sorry, I'm having trouble right now. Could you please try again?"


async def generate_response(user_text: str) -> str:
    try:
        response = await client.chat.completions.create(
            model=VOICE_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful voice assistant. "
                        "Keep ALL responses under 2-3 short sentences — you are speaking aloud, not writing. "
                        "Never use bullet points, headers, or markdown."
                    )
                },
                {
                    "role": "user",
                    "content": user_text
                }
            ],
            **_VOICE_REASONING_KW,
            temperature=0.7,
            max_tokens=200,
        )
    except Exception as exc:
        # 429/413/timeout/network — degrade gracefully instead of a 500.
        logger.warning(f"generate_response: Groq call failed ({exc}); returning fallback")
        return _FALLBACK

    # A gpt-oss reasoning model can spend its whole budget on hidden reasoning and
    # return content=None. Guard it so we never hand None to TTS (→ Sarvam 400).
    content = (response.choices[0].message.content or "").strip()
    return content or _FALLBACK