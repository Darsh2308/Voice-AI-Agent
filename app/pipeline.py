from app.asr import transcribe_audio
from app.llm import generate_response
from app.tts import text_to_speech

async def run_pipeline(audio_path: str):
    print("1. Transcribing...")
    text = await transcribe_audio(audio_path)

    print("User said:", text)

    print("2. Generating response...")
    response = await generate_response(text)

    print("AI response:", response)

    print("3. Converting to speech...")
    output_audio = await text_to_speech(response)

    print("Output saved at:", output_audio)

    return output_audio