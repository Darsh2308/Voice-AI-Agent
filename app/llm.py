from groq import AsyncGroq
from loguru import logger
from app.config import GROQ_API_KEY, VOICE_LLM_MODEL

# Legacy one-shot helper for the POST /voice REST endpoint (NOT the production
# /ws streaming path). A 15s timeout keeps a stuck request from hanging the
# handler far longer than a phone caller would wait.
client = AsyncGroq(api_key=GROQ_API_KEY, timeout=15.0)

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
            reasoning_effort="low",   # GPT-OSS reasoning model — minimise overhead
            temperature=0.7,
            max_tokens=200,   # raised for reasoning model headroom
        )
    except Exception as exc:
        # 429/413/timeout/network — degrade gracefully instead of a 500.
        logger.warning(f"generate_response: Groq call failed ({exc}); returning fallback")
        return _FALLBACK

    # A gpt-oss reasoning model can spend its whole budget on hidden reasoning and
    # return content=None. Guard it so we never hand None to TTS (→ Sarvam 400).
    content = (response.choices[0].message.content or "").strip()
    return content or _FALLBACK