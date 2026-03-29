from groq import AsyncGroq
from app.config import GROQ_API_KEY

client = AsyncGroq(api_key=GROQ_API_KEY)

async def generate_response(user_text: str) -> str:
    response = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
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
        temperature=0.7,
        max_tokens=120,   # ~80–100 spoken words — fits comfortably in Sarvam TTS
    )

    return response.choices[0].message.content