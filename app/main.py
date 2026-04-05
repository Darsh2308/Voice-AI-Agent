"""
main.py — WebSocket server with all enhanced features
=======================================================

Changes from the original Phase 7 version:

Feature 1  – Interrupt Handling Fix
  After manager.interrupt() is called, the output_queue is already drained
  inside VoicePipelineManager. Here we set is_ai_speaking=False and
  last_ai_finished_at=0.0 immediately so receive_loop stops blocking audio.

Feature 2  – Connection Timeout
  timeout_watch() coroutine checks inactivity every 60 seconds. If no audio
  has arrived for 10 minutes it sends a {"type":"timeout"} message, closes
  the WebSocket, and lets the finally block clean up the pipeline.

Feature 4  – Barge-in
  Echo cooldown is reduced to 0.3 s (WebRTC echo cancellation in the browser
  handles the rest). Audio is ALWAYS forwarded to the pipeline — including
  while the AI is speaking. When VAD detects speech during AI playback it
  emits BargeInDetectedFrame, which send_loop catches to auto-interrupt.
  manager.set_barge_in_mode() tells the VAD when to watch for barge-in.

Feature 6  – Typing Indicator
  AIThinkingFrame(True/False) from GroqLangGraphProcessor is forwarded to the
  browser as {"type":"thinking","active":true/false}.

Feature 7  – Language Auto-Switch
  LanguageDetectedFrame from SarvamSTTService is forwarded to the browser as
  {"type":"language","code":"hi-IN"} so the UI can show a language badge.
"""

import asyncio
import json
import os
import sys
import time

from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.responses import FileResponse
from loguru import logger
import shutil

logger.remove()
logger.add(sys.stdout, level="DEBUG")

from app.pipeline import run_pipeline
from app.pipecat_pipeline import (
    VoicePipelineManager,
    AIAudioFrame,
    AIStatusFrame,
    AIThinkingFrame,          # Feature 6
    LanguageDetectedFrame,    # Feature 7
    BargeInDetectedFrame,     # Feature 4
    TranscriptDisplayFrame,
    EndFrame,
)

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Voice AI Agent Running 🚀"}


@app.post("/voice")
async def voice_agent(file: UploadFile = File(...)):
    """One-shot voice endpoint — unchanged."""
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    output_audio = await run_pipeline(file_path)
    return {"message": "Processed successfully", "output_audio": output_audio}


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_BROWSER_SAMPLE_RATE = 48000

# Feature 4: reduced from 1.5 s to 0.3 s because WebRTC echo cancellation
# (enabled in the browser) handles the bulk of mic-pickup-of-speaker echo.
POST_AI_COOLDOWN_SECS = 0.3

# Feature 2: close the connection after this many seconds of silence
INACTIVITY_TIMEOUT_SECS = 600   # 10 minutes


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Full-duplex real-time voice agent.

    Three coroutines run concurrently (asyncio.gather):
      receive_loop()   — reads PCM audio + control messages from browser
      send_loop()      — reads pipeline output frames, sends to browser
      timeout_watch()  — closes idle connections after INACTIVITY_TIMEOUT_SECS
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    # ── Per-connection state ──────────────────────────────────────────────────
    browser_sample_rate = DEFAULT_BROWSER_SAMPLE_RATE
    # Feature 4: last_ai_finished_at starts at 0.0 so elapsed is huge and audio
    # flows immediately (no cooldown at session start).
    last_ai_finished_at = 0.0
    is_ai_speaking      = False
    # Feature 2: track last audio arrival for inactivity detection
    last_activity       = time.time()

    manager = VoicePipelineManager()
    await manager.start()

    # ── receive_loop ─────────────────────────────────────────────────────────
    async def receive_loop():
        """
        Read frames from the browser:
          - Binary frames  → raw PCM audio → pushed into pipeline after cooldown
          - Text frames    → JSON control messages (init, interrupt)

        Feature 4 (Barge-in): audio is forwarded to the pipeline even while
        the AI is speaking. The VAD is in barge-in mode during that time and
        will emit BargeInDetectedFrame when it detects the user speaking,
        which send_loop catches and uses to interrupt.

        The cooldown (0.3 s) only prevents the very first chunks after the AI
        stops speaking from being picked up (residual speaker echo). WebRTC echo
        cancellation in the browser handles the rest.
        """
        nonlocal browser_sample_rate, last_ai_finished_at, last_activity, is_ai_speaking

        try:
            while True:
                message = await websocket.receive()

                # ── Text frame: control message ───────────────────────────
                if "text" in message:
                    try:
                        meta = json.loads(message["text"])

                        if meta.get("type") == "init":
                            browser_sample_rate = int(meta["sampleRate"])
                            manager.update_sample_rate(browser_sample_rate)
                            logger.info(f"Browser sample rate: {browser_sample_rate} Hz")

                        elif meta.get("type") == "interrupt":
                            # Feature 1: interrupt + drain stale frames
                            await manager.interrupt()
                            is_ai_speaking      = False
                            last_ai_finished_at = 0.0    # no cooldown after interrupt
                            manager.set_barge_in_mode(False)   # Feature 4
                            await websocket.send_text(json.dumps({"type": "interrupted"}))
                            logger.info("Manual interrupt received from client")

                    except (json.JSONDecodeError, KeyError):
                        pass
                    continue

                # ── Binary frame: raw PCM audio ───────────────────────────
                pcm_bytes = message.get("bytes", b"")
                if not pcm_bytes:
                    continue

                # Feature 2: record activity time
                last_activity = time.time()

                # Feature 4 (Barge-in) + original echo cooldown:
                # Apply a SHORT cooldown (0.3 s) only after the AI finishes a
                # sentence. During this tiny window the last few frames of
                # speaker audio may not yet be cancelled by WebRTC EC.
                # Outside this window, audio always flows — even during AI speech
                # (that's what enables barge-in).
                elapsed = time.time() - last_ai_finished_at
                if elapsed < POST_AI_COOLDOWN_SECS:
                    continue

                await manager.push_audio(pcm_bytes, sample_rate=browser_sample_rate)

        except Exception as e:
            logger.info(f"receive_loop ended: {e}")

    # ── send_loop ─────────────────────────────────────────────────────────────
    async def send_loop():
        """
        Read output frames from the pipeline and send them to the browser.

        Frame types:
          TranscriptDisplayFrame  → "User: …" / "AI: …" chat lines
          AIStatusFrame           → {"type":"status","ai_speaking":bool}
          AIAudioFrame            → raw WAV bytes for playback
          AIThinkingFrame         → Feature 6: {"type":"thinking","active":bool}
          LanguageDetectedFrame   → Feature 7: {"type":"language","code":"hi-IN"}
          BargeInDetectedFrame    → Feature 4: auto-interrupt during playback
          EndFrame                → pipeline shutdown — exit loop
        """
        nonlocal last_ai_finished_at, is_ai_speaking

        while True:
            try:
                frame = await asyncio.wait_for(
                    manager.output_queue.get(),
                    timeout=60.0,
                )
            except asyncio.TimeoutError:
                continue

            # ── TranscriptDisplayFrame ────────────────────────────────────
            if isinstance(frame, TranscriptDisplayFrame):
                prefix = "User" if frame.speaker == "user" else "AI"
                await websocket.send_text(f"{prefix}: {frame.text}")

            # ── AIStatusFrame ─────────────────────────────────────────────
            elif isinstance(frame, AIStatusFrame):
                if frame.ai_speaking and not is_ai_speaking:
                    # AI just started speaking — enable barge-in mode on VAD
                    is_ai_speaking = True
                    manager.set_barge_in_mode(True)   # Feature 4

                elif not frame.ai_speaking and is_ai_speaking:
                    # AI finished a sentence — start brief echo cooldown
                    is_ai_speaking      = False
                    last_ai_finished_at = time.time()
                    manager.set_barge_in_mode(False)  # Feature 4

                await websocket.send_text(
                    json.dumps({"type": "status", "ai_speaking": frame.ai_speaking})
                )

            # ── AIAudioFrame ──────────────────────────────────────────────
            elif isinstance(frame, AIAudioFrame):
                # Feature 3: multiple WAV chunks arrive (one per sentence).
                # Browser queues and plays them sequentially.
                await websocket.send_bytes(frame.audio_bytes)
                logger.info(f"Sent {len(frame.audio_bytes)} audio bytes to client")

            # ── Feature 6: AIThinkingFrame ────────────────────────────────
            elif isinstance(frame, AIThinkingFrame):
                await websocket.send_text(
                    json.dumps({"type": "thinking", "active": frame.thinking})
                )

            # ── Feature 7: LanguageDetectedFrame ─────────────────────────
            elif isinstance(frame, LanguageDetectedFrame):
                await websocket.send_text(
                    json.dumps({"type": "language", "code": frame.language_code})
                )

            # ── Feature 4: BargeInDetectedFrame ──────────────────────────
            elif isinstance(frame, BargeInDetectedFrame):
                if is_ai_speaking:
                    logger.info("Barge-in detected — interrupting AI")
                    # Feature 1: interrupt + drain stale frames from queue
                    await manager.interrupt()
                    is_ai_speaking      = False
                    last_ai_finished_at = 0.0    # no cooldown — user is already speaking
                    manager.set_barge_in_mode(False)
                    # Tell browser to stop current audio and show "Listening"
                    await websocket.send_text(json.dumps({"type": "barge_in"}))

            # ── EndFrame ──────────────────────────────────────────────────
            elif isinstance(frame, EndFrame):
                logger.info("Pipeline EndFrame received — send_loop exiting")
                break

    # ── Feature 2: timeout_watch ─────────────────────────────────────────────
    async def timeout_watch():
        """
        Feature 2: Close idle WebSocket connections after 10 minutes.
        Checks every 60 seconds whether any audio has arrived recently.
        Sends a {"type":"timeout"} message before closing so the browser
        can display a friendly "session ended" message.
        """
        while True:
            await asyncio.sleep(60)
            idle_secs = time.time() - last_activity
            if idle_secs >= INACTIVITY_TIMEOUT_SECS:
                logger.info(f"Connection idle for {idle_secs:.0f}s — closing")
                try:
                    await websocket.send_text(json.dumps({"type": "timeout"}))
                    await websocket.close()
                except Exception:
                    pass
                return

    # ── Run all three coroutines concurrently ─────────────────────────────────
    try:
        await asyncio.gather(receive_loop(), send_loop(), timeout_watch())
    except Exception as e:
        logger.info(f"WebSocket session ended: {e}")
    finally:
        await manager.stop()
        logger.info("WebSocket client disconnected — pipeline cleaned up")
