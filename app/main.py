"""
Phase 7 – main.py  (simplified WebSocket using Pipecat pipeline)
=================================================================

WHAT CHANGED FROM PHASE 6
──────────────────────────
Phase 6 main.py had ~280 lines. The WebSocket handler manually:
  - maintained the VAD state machine
  - called ASR, LLM, TTS in sequence
  - tracked is_ai_speaking, echo cooldown, chunk counters
  - created asyncio Tasks for each utterance

Phase 7 main.py has ~80 lines of actual logic. The WebSocket handler:
  - creates a VoicePipelineManager (the pipeline lives in pipecat_pipeline.py)
  - runs two simple coroutines: one that SENDS audio in, one that READS output out
  - handles echo cooldown (prevents AI's own voice from being re-transcribed)
  - handles manual interrupts from the browser ⚡ button

All the hard work (VAD, ASR, LLM, TTS, memory) is now in the pipeline.
main.py is pure I/O routing.
"""

import asyncio
import json
import time

from fastapi import FastAPI, File, UploadFile, WebSocket
from loguru import logger
import shutil

from app.pipeline import run_pipeline                  # keeps the POST /voice endpoint working
from app.pipecat_pipeline import (
    VoicePipelineManager,
    AIAudioFrame,
    AIStatusFrame,
    TranscriptDisplayFrame,
    EndFrame,
)

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Voice AI Agent Running (Phase 7 – Pipecat) 🚀"}


@app.post("/voice")
async def voice_agent(file: UploadFile = File(...)):
    """
    Simple one-shot voice endpoint — unchanged from Phase 6.
    Upload a WAV, get a response WAV path back.
    Used for testing outside of WebSocket flow.
    """
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    output_audio = await run_pipeline(file_path)
    return {"message": "Processed successfully", "output_audio": output_audio}


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_BROWSER_SAMPLE_RATE = 48000   # fallback if init message not received
POST_AI_COOLDOWN_SECS       = 1.5    # seconds to ignore audio after AI speaks
                                      # (prevents AI's own voice from being picked up)


# ── WebSocket endpoint (Phase 7: Pipecat-based) ───────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Full-duplex real-time voice agent using Pipecat pipeline.

    Two coroutines run concurrently:

      receive_loop()  — reads binary PCM frames from the browser and feeds
                        them into the pipeline. Also handles text control
                        messages (init metadata, interrupt button).

      send_loop()     — reads output frames from the pipeline's output queue
                        and sends them back to the browser (text transcripts,
                        WAV audio, speaking-status JSON).

    The pipeline (VAD → STT → LLM → TTS) runs as a separate background task
    managed by VoicePipelineManager.
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    # ── Per-connection state ──────────────────────────────────────────────────
    browser_sample_rate = DEFAULT_BROWSER_SAMPLE_RATE
    last_ai_finished_at = 0.0       # timestamp when AI last finished speaking
    is_ai_speaking      = False     # True while send_loop is sending AI audio

    # ── Start the Pipecat pipeline ────────────────────────────────────────────
    # Each connection gets its own VoicePipelineManager instance, which means
    # each connection has its own conversation history (memory is per-session).
    manager = VoicePipelineManager()
    await manager.start()

    # ── receive_loop ─────────────────────────────────────────────────────────
    async def receive_loop():
        """
        Read frames from the browser and route them:
          - Binary frames  → raw PCM audio → push into pipeline
          - Text frames    → JSON control messages (init, interrupt)

        Also applies the echo cooldown: if AI finished speaking less than
        POST_AI_COOLDOWN_SECS ago, we discard incoming audio to avoid
        re-transcribing the AI's own voice picked up by the mic.
        """
        nonlocal browser_sample_rate, last_ai_finished_at

        try:
            while True:
                message = await websocket.receive()

                # ── Text frame: control message ───────────────────────────
                if "text" in message:
                    try:
                        meta = json.loads(message["text"])

                        if meta.get("type") == "init":
                            # Browser tells us its native sample rate
                            browser_sample_rate = int(meta["sampleRate"])
                            manager.update_sample_rate(browser_sample_rate)
                            logger.info(f"Browser sample rate: {browser_sample_rate} Hz")

                        elif meta.get("type") == "interrupt":
                            # User pressed the ⚡ Interrupt AI button.
                            # 1. Cancel pipeline processing on the server.
                            # 2. Tell the browser to stop local audio playback too.
                            await manager.interrupt()
                            await websocket.send_text(json.dumps({"type": "interrupted"}))
                            logger.info("Manual interrupt received from client")

                    except (json.JSONDecodeError, KeyError):
                        pass
                    continue

                # ── Binary frame: raw PCM audio ───────────────────────────
                pcm_bytes = message.get("bytes", b"")
                if not pcm_bytes:
                    continue

                # ── Echo cooldown guard ───────────────────────────────────
                # Prevent the AI's own TTS audio (picked up by the mic) from
                # flowing into the pipeline and being transcribed again.
                elapsed = time.time() - last_ai_finished_at
                if elapsed < POST_AI_COOLDOWN_SECS:
                    continue   # discard this chunk silently

                # ── Feed audio into the Pipecat pipeline ──────────────────
                # VADProcessor (inside the pipeline) handles all the logic:
                # buffering, RMS thresholding, speech/silence detection.
                # We just push raw chunks and the pipeline does the rest.
                await manager.push_audio(pcm_bytes, sample_rate=browser_sample_rate)

        except Exception as e:
            logger.info(f"receive_loop ended: {e}")

    # ── send_loop ─────────────────────────────────────────────────────────────
    async def send_loop():
        """
        Read output frames from the pipeline and send them to the browser.

        Frame types we handle:
          TranscriptDisplayFrame → "User: …" or "AI: …" text lines for chat UI
          AIStatusFrame          → JSON {"type":"status","ai_speaking":true/false}
          AIAudioFrame           → raw WAV bytes for browser audio playback
          EndFrame               → pipeline shut down, exit the loop
        """
        nonlocal last_ai_finished_at, is_ai_speaking

        while True:
            try:
                # Block until the pipeline produces something
                frame = await asyncio.wait_for(
                    manager.output_queue.get(),
                    timeout=60.0,   # safety timeout per get()
                )
            except asyncio.TimeoutError:
                continue

            # ── TranscriptDisplayFrame ────────────────────────────────────
            if isinstance(frame, TranscriptDisplayFrame):
                # Shows up in the browser's chat-style conversation panel
                prefix = "User" if frame.speaker == "user" else "AI"
                await websocket.send_text(f"{prefix}: {frame.text}")

            # ── AIStatusFrame ─────────────────────────────────────────────
            elif isinstance(frame, AIStatusFrame):
                # Drives the browser's speaking indicator (red pulse animation)
                is_ai_speaking = frame.ai_speaking
                if not frame.ai_speaking:
                    # AI just finished speaking — start the echo cooldown clock
                    last_ai_finished_at = time.time()
                await websocket.send_text(
                    json.dumps({"type": "status", "ai_speaking": frame.ai_speaking})
                )

            # ── AIAudioFrame ──────────────────────────────────────────────
            elif isinstance(frame, AIAudioFrame):
                # Send synthesized WAV bytes to the browser for playback
                await websocket.send_bytes(frame.audio_bytes)
                logger.info(f"Sent {len(frame.audio_bytes)} audio bytes to client")

            # ── EndFrame ──────────────────────────────────────────────────
            elif isinstance(frame, EndFrame):
                # Pipeline shut down (connection closing) — exit the loop
                logger.info("Pipeline EndFrame received — send_loop exiting")
                break

    # ── Run both loops concurrently ───────────────────────────────────────────
    # asyncio.gather() runs receive_loop and send_loop at the same time.
    # When either raises an exception (e.g. WebSocket disconnect), both stop.
    try:
        await asyncio.gather(receive_loop(), send_loop())
    except Exception as e:
        logger.info(f"WebSocket session ended: {e}")
    finally:
        await manager.stop()
        logger.info("WebSocket client disconnected — pipeline cleaned up")
