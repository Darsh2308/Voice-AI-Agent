"""
Phase 7 – Pipecat Pipeline Architecture
=========================================

WHY THIS FILE EXISTS
────────────────────
Phase 6 crammed ALL orchestration into one 280-line WebSocket handler:
  - VAD state machine (RMS thresholds, chunk counters)
  - resampling
  - ASR → LLM → TTS chaining
  - echo-cooldown timer
  - interrupt / cancellation logic

That works, but it's monolithic. Any change (swap TTS provider, add
translation, tune VAD) means editing one giant function.

Pipecat solves this with a FRAME-BASED PIPELINE:
  - Every piece of data (audio chunk, transcript, AI response, audio bytes)
    is wrapped in a typed "Frame" object.
  - Each processing stage (VAD, STT, LLM, TTS) is a standalone
    FrameProcessor class that receives frames, does its work, and pushes
    new frames downstream.
  - A Pipeline wires the processors together. A PipelineTask runs
    everything asynchronously.

FRAME FLOW (Phase 7)
──────────────────────────────────────────────────────────────
  WebSocket binary
        │  AudioRawFrame (raw 16-bit PCM from browser)
        ▼
  ┌─────────────────┐
  │  VADProcessor   │  Voice Activity Detection.
  │                 │  Buffers audio, detects speech start/end,
  │                 │  emits one SpeechEndFrame per utterance.
  └────────┬────────┘
           │  SpeechEndFrame (buffered WAV bytes of one utterance)
           ▼
  ┌──────────────────────┐
  │  SarvamSTTService    │  Calls Sarvam ASR API.
  │                      │  Emits TranscriptionFrame (for LLM)
  │                      │  and TranscriptDisplayFrame (for browser chat UI).
  └──────────┬───────────┘
             │  TranscriptionFrame (user's words as text)
             ▼
  ┌──────────────────────┐
  │ GroqLangGraphProcessor│ Phase 8: LangGraph state graph.
  │                      │  run_agent(text, thread_id) is called per turn.
  │                      │  MemorySaver stores full history per session.
  │                      │  Emits TextFrame (AI reply)
  │                      │  and TranscriptDisplayFrame (for browser).
  └──────────┬───────────┘
             │  TextFrame (AI's response as text)
             ▼
  ┌──────────────────────┐
  │  SarvamTTSService    │  Calls Sarvam TTS API.
  │                      │  Emits AIAudioFrame (WAV bytes to send to browser).
  └──────────┬───────────┘
             │  AIAudioFrame (synthesized speech as WAV bytes)
             ▼
  ┌──────────────────────┐
  │  OutputSink          │  Puts frames onto an asyncio.Queue.
  │                      │  main.py reads from this queue to send
  │                      │  audio/text back to the browser.
  └──────────────────────┘

KEY IMPROVEMENT OVER PHASE 6
──────────────────────────────
  Phase 6: Every response had NO memory of previous turns.
           Each call to generate_response() was a fresh one-shot prompt.

  Phase 7: GroqLLMProcessor maintains a running conversation history.
           The AI now remembers what was said earlier in the session!
"""

import array
import asyncio
import base64
import io
import os
import tempfile
import time
import uuid
import wave
from typing import List, Optional

import httpx
from loguru import logger

# ─────────────────────────────────────────────────────────────────────────────
# Pipecat Core Imports
# ─────────────────────────────────────────────────────────────────────────────
#
# Frame       – base class for every data packet in the pipeline
# AudioRawFrame – carries raw PCM audio bytes
# TextFrame   – carries text (LLM response tokens)
# TranscriptionFrame – ASR output; the built-in type the LLM context expects
# EndFrame    – signals the pipeline to shut down gracefully
#
from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    Frame,
    TextFrame,
    TranscriptionFrame,
)

# FrameProcessor – base class for all pipeline stages
# FrameDirection – DOWNSTREAM (towards output) or UPSTREAM (towards input)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# Pipeline   – wires processors together in order
# PipelineRunner – drives the async execution of a PipelineTask
# PipelineTask   – a single running instance of a Pipeline
# PipelineParams – configuration knobs for the task
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

from app.config import GROQ_API_KEY, SARVAM_API_KEY


# ─────────────────────────────────────────────────────────────────────────────
# Custom Frame Types
# ─────────────────────────────────────────────────────────────────────────────
#
# Pipecat's built-in frames (AudioRawFrame, TextFrame, …) cover common cases.
# We define our own for project-specific data that doesn't fit those types.
#

class SpeechEndFrame(Frame):
    """
    Emitted by VADProcessor when a full utterance has been detected.
    Carries the complete buffered WAV audio (already resampled to 16 kHz)
    ready to be sent to Sarvam ASR.
    """
    def __init__(self, audio_bytes: bytes, sample_rate: int = 16000):
        super().__init__()
        self.audio_bytes = audio_bytes          # WAV file bytes
        self.sample_rate = sample_rate          # always 16 kHz after resampling


class TranscriptDisplayFrame(Frame):
    """
    Emitted by STT (for user text) and LLM (for AI text).
    Carries a line of conversation text that the browser chat UI should display.
    NOT used for LLM input — that's TranscriptionFrame.
    """
    def __init__(self, text: str, speaker: str = "user"):
        super().__init__()
        self.text = text            # the message
        self.speaker = speaker      # "user" or "ai"


class AIAudioFrame(Frame):
    """
    Emitted by SarvamTTSService.
    Carries synthesized WAV audio bytes to be sent to the browser for playback.
    """
    def __init__(self, audio_bytes: bytes):
        super().__init__()
        self.audio_bytes = audio_bytes      # raw WAV file bytes


class AIStatusFrame(Frame):
    """
    Emitted by SarvamTTSService before and after generating audio.
    Lets main.py track whether the AI is currently speaking.
    """
    def __init__(self, ai_speaking: bool):
        super().__init__()
        self.ai_speaking = ai_speaking      # True = started, False = finished


# ─────────────────────────────────────────────────────────────────────────────
# 1.  VADProcessor  (Voice Activity Detection)
# ─────────────────────────────────────────────────────────────────────────────
#
# Replaces the raw VAD state machine that lived inside the Phase 6
# WebSocket handler. Same algorithm, but now it's a self-contained
# FrameProcessor that can be swapped out independently.
#
# Receives:  AudioRawFrame  (one ~85ms PCM chunk from the browser)
# Emits:     SpeechEndFrame (one complete utterance as WAV)
#

class VADProcessor(FrameProcessor):
    """
    Voice Activity Detection – detects when the user starts and stops speaking,
    then emits a single SpeechEndFrame containing the full buffered utterance.

    Algorithm (same as Phase 6):
      1. Compute RMS energy of each incoming PCM chunk.
      2. If energy > SPEECH_RMS_THRESHOLD for MIN_SPEECH_CHUNKS consecutive
         chunks → speech started, begin buffering.
      3. Once speech started, if energy < SPEECH_RMS_THRESHOLD for
         SILENCE_CHUNKS_NEEDED consecutive chunks → utterance complete.
      4. Hard-cap at MAX_BUFFER_CHUNKS to avoid infinite wait.
    """

    # ── Tunable thresholds (identical to Phase 6 values) ──────────────────────
    SPEECH_RMS_THRESHOLD = 700      # below → silence; above → speech
    MIN_SPEECH_CHUNKS    = 6        # need ~0.5 s of real speech before buffering
    SILENCE_CHUNKS_NEEDED = 8       # ~0.67 s of quiet means utterance ended
    MAX_BUFFER_CHUNKS    = 80       # ~6.7 s safety cap

    # ── Sample rate constants ─────────────────────────────────────────────────
    TARGET_SAMPLE_RATE   = 16000    # Sarvam ASR expects 16 kHz

    def __init__(self, browser_sample_rate: int = 48000, **kwargs):
        super().__init__(**kwargs)
        # The browser sends audio at its native rate (usually 48 kHz).
        # We need to resample to 16 kHz before storing in the buffer.
        self._browser_rate = browser_sample_rate
        self._reset_vad_state()

    def update_sample_rate(self, rate: int):
        """Called by main.py once the browser sends its init metadata."""
        self._browser_rate = rate
        logger.info(f"VAD: browser sample rate updated to {rate} Hz")

    def _reset_vad_state(self):
        """Clear all VAD buffers after emitting an utterance."""
        self._audio_buffer: List[bytes] = []    # resampled 16-kHz PCM chunks
        self._speech_chunks_seen  = 0
        self._silence_chunk_count = 0
        self._is_speech_active    = False

    # ── RMS calculation ───────────────────────────────────────────────────────

    def _rms(self, pcm_bytes: bytes) -> float:
        """
        Root Mean Square of 16-bit signed little-endian PCM data.
        Higher value = louder audio. Used to classify speech vs. silence.
        """
        if len(pcm_bytes) < 2:
            return 0.0
        # Unpack raw bytes as signed 16-bit integers
        samples = array.array('h', pcm_bytes[:len(pcm_bytes) & ~1])
        if not samples:
            return 0.0
        return (sum(s * s for s in samples) / len(samples)) ** 0.5

    # ── Resampling ────────────────────────────────────────────────────────────

    def _resample(self, pcm_bytes: bytes) -> bytes:
        """
        Downsample from browser rate to 16 kHz using simple decimation.
        Works correctly when the ratio is an integer (48k/16k = 3).
        """
        if self._browser_rate == self.TARGET_SAMPLE_RATE:
            return pcm_bytes
        ratio = self._browser_rate / self.TARGET_SAMPLE_RATE   # e.g. 3.0
        samples = array.array('h', pcm_bytes[:len(pcm_bytes) & ~1])
        out_len = int(len(samples) / ratio)
        out = array.array('h', (samples[int(i * ratio)] for i in range(out_len)))
        return out.tobytes()

    # ── WAV packing ──────────────────────────────────────────────────────────

    def _pcm_to_wav(self, pcm_bytes: bytes) -> bytes:
        """
        Wrap raw PCM bytes in a WAV header.
        Sarvam ASR expects a proper WAV file, not bare PCM.
        """
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)          # mono
            wf.setsampwidth(2)          # 16-bit
            wf.setframerate(self.TARGET_SAMPLE_RATE)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    # ── Pipecat frame processing ──────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        Called by the Pipecat pipeline for every frame that arrives here.

        We only act on AudioRawFrame; everything else passes through
        unchanged so downstream processors can see it.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            await self._process_audio_chunk(frame.audio)
        else:
            # Not our frame type — pass it downstream unchanged
            await self.push_frame(frame, direction)

    async def _process_audio_chunk(self, raw_pcm: bytes):
        """Run one chunk of browser audio through the VAD state machine."""

        # Step 1: resample to 16 kHz
        pcm_16k = self._resample(raw_pcm)

        # Step 2: compute energy of the 16-kHz chunk
        energy = self._rms(pcm_16k)

        if energy >= self.SPEECH_RMS_THRESHOLD:
            # ── SPEECH chunk ─────────────────────────────────────────────────
            self._silence_chunk_count = 0
            self._audio_buffer.append(pcm_16k)

            if not self._is_speech_active:
                self._speech_chunks_seen += 1
                if self._speech_chunks_seen >= self.MIN_SPEECH_CHUNKS:
                    self._is_speech_active = True
                    logger.debug("VAD: speech STARTED")
        else:
            # ── SILENCE chunk ─────────────────────────────────────────────────
            if self._is_speech_active:
                # Keep buffering (trailing silence is part of the utterance)
                self._silence_chunk_count += 1
                self._audio_buffer.append(pcm_16k)

                silence_ended  = self._silence_chunk_count >= self.SILENCE_CHUNKS_NEEDED
                hard_cap_hit   = len(self._audio_buffer) >= self.MAX_BUFFER_CHUNKS

                if silence_ended or hard_cap_hit:
                    reason = "silence" if silence_ended else "hard-cap"
                    logger.info(f"VAD: utterance END ({reason})")
                    await self._emit_utterance()

    async def _emit_utterance(self):
        """Package buffered audio into WAV and push a SpeechEndFrame downstream."""
        if not self._audio_buffer:
            self._reset_vad_state()
            return

        raw_pcm  = b"".join(self._audio_buffer)
        wav_bytes = self._pcm_to_wav(raw_pcm)

        logger.info(f"VAD: emitting SpeechEndFrame — {len(raw_pcm)} bytes PCM → {len(wav_bytes)} bytes WAV")

        # Push the frame to the next stage (SarvamSTTService)
        await self.push_frame(SpeechEndFrame(audio_bytes=wav_bytes, sample_rate=self.TARGET_SAMPLE_RATE))

        self._reset_vad_state()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SarvamSTTService  (Speech-to-Text)
# ─────────────────────────────────────────────────────────────────────────────
#
# Receives:  SpeechEndFrame  (WAV audio of one utterance)
# Emits:     TranscriptionFrame      (text → goes to GroqLLMProcessor)
#            TranscriptDisplayFrame  (text → goes to browser chat UI)
#

class SarvamSTTService(FrameProcessor):
    """
    Calls the Sarvam ASR API to transcribe one complete utterance.

    Why custom? Pipecat has built-in STT services for Deepgram, AssemblyAI,
    etc., but not for Sarvam. Writing a custom FrameProcessor is the correct
    way to plug in a provider that Pipecat doesn't know about.
    """

    SARVAM_ASR_URL = "https://api.sarvam.ai/speech-to-text"

    # Single-word filler responses that are almost certainly ASR noise
    FILLER_WORDS = {
        "yes", "no", "ok", "okay", "hmm", "uh", "um", "ah", "oh",
        "huh", "hm", "yeah", "yep", "nope", "hey"
    }

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self._api_key    = api_key
        # Reuse one HTTP client for the lifetime of this processor
        self._http = httpx.AsyncClient(timeout=30.0)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, SpeechEndFrame):
            await self._transcribe(frame)
        else:
            await self.push_frame(frame, direction)

    async def _transcribe(self, frame: SpeechEndFrame):
        """
        Upload the WAV audio to Sarvam ASR and emit the transcript.

        Sarvam expects multipart/form-data, so we write a temporary file
        (same pattern as the original asr.py).
        """
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="pipecat_utt_")
        try:
            # Write WAV bytes to a temp file for the multipart upload
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(frame.audio_bytes)

            with open(tmp_path, "rb") as f:
                resp = await self._http.post(
                    self.SARVAM_ASR_URL,
                    headers={"api-subscription-key": self._api_key},
                    files={"file": ("audio.wav", f, "audio/wav")},
                    data={"model": "saarika:v2.5", "language_code": "en-IN"},
                )

            if resp.status_code != 200:
                logger.error(f"STT HTTP {resp.status_code}: {resp.text[:200]}")
                return

            transcript = resp.json().get("transcript", "").strip()
            logger.info(f"STT transcript: {transcript!r}")

            # ── Noise / filler filter ──────────────────────────────────────
            cleaned = transcript.lower().rstrip(".,!? ")
            if not cleaned or len(cleaned) <= 2 or cleaned in self.FILLER_WORDS:
                logger.debug(f"STT: filtered noise/filler {transcript!r}")
                return

            # ── Emit frames downstream ─────────────────────────────────────

            # TranscriptionFrame — the standard pipecat frame for ASR output.
            # GroqLLMProcessor listens for this type.
            await self.push_frame(
                TranscriptionFrame(text=transcript, user_id="user", timestamp="")
            )

            # TranscriptDisplayFrame — carries the same text to the browser UI.
            # It flows all the way to OutputSink which puts it on the output queue.
            await self.push_frame(
                TranscriptDisplayFrame(text=transcript, speaker="user")
            )

        except Exception as e:
            logger.error(f"STT error: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    async def cleanup(self):
        await self._http.aclose()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GroqLangGraphProcessor  (Phase 8 LLM + Memory via LangGraph)
# ─────────────────────────────────────────────────────────────────────────────
#
# Receives:  TranscriptionFrame  (user's transcribed speech)
# Emits:     TextFrame               (AI's reply → goes to TTS)
#            TranscriptDisplayFrame  (AI's reply → goes to browser chat UI)
#
# PHASE 8 UPGRADE vs PHASE 7:
#   Phase 7: self._messages = [dict, dict, ...]  → manual list append/pop
#             Direct Groq httpx call
#             Memory lives only in the object
#
#   Phase 8: LangGraph StateGraph with add_messages reducer
#             MemorySaver checkpointer with thread_id isolation
#             Memory survives object replacement (stored in checkpointer)
#             Foundation for adding tools, routing, multi-agent, etc.
#
# WHY thread_id?
#   The same `agent_graph` (compiled LangGraph) is shared across all
#   WebSocket connections. The thread_id is the key that tells LangGraph
#   "this turn belongs to connection X, load that conversation history".
#   Without it, all users would share the same memory!
#

class GroqLangGraphProcessor(FrameProcessor):
    """
    Phase 8 LLM processor: delegates all reasoning and memory to LangGraph.

    Each instance has a unique thread_id (UUID) assigned by VoicePipelineManager.
    When process_frame() receives a TranscriptionFrame, it calls run_agent()
    which routes through the LangGraph state machine with automatic memory.
    """

    def __init__(self, thread_id: str, **kwargs):
        super().__init__(**kwargs)
        # Unique identifier for this session's conversation history.
        # LangGraph uses this to load/save the right state from MemorySaver.
        self._thread_id = thread_id
        logger.info(f"GroqLangGraphProcessor: thread_id={thread_id}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            await self._generate(frame.text)
        else:
            await self.push_frame(frame, direction)

    async def _generate(self, user_text: str):
        """
        Route user text through the LangGraph agent and emit the response.

        What happens inside run_agent():
          1. LangGraph loads saved state for self._thread_id from MemorySaver
          2. Appends HumanMessage(user_text) to history (via add_messages)
          3. Calls llm_node → Groq API with FULL history
          4. Appends AIMessage(reply) to history
          5. Saves state back to MemorySaver
          6. Returns the AI's text reply
        """
        try:
            # Import here to avoid circular import at module load time
            from app.langgraph_flow import run_agent

            logger.info(f"LangGraph: turn for thread={self._thread_id[:8]}… input={user_text!r}")

            ai_reply = await run_agent(user_text, self._thread_id)

            if not ai_reply:
                logger.warning("LangGraph: empty reply, skipping")
                return

            # Push text downstream → SarvamTTSService will speak it
            await self.push_frame(TextFrame(text=ai_reply))

            # Push for browser chat display
            await self.push_frame(TranscriptDisplayFrame(text=ai_reply, speaker="ai"))

        except Exception as e:
            logger.error(f"LangGraph error: {e}")

    def reset_thread(self):
        """
        Start a new conversation by assigning a new thread_id.
        The old history is not deleted from MemorySaver, just orphaned.
        A fresh UUID means the next turn starts with an empty history.
        """
        old = self._thread_id
        self._thread_id = str(uuid.uuid4())
        logger.info(f"LangGraph: thread reset {old[:8]}→{self._thread_id[:8]}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SarvamTTSService  (Text-to-Speech)
# ─────────────────────────────────────────────────────────────────────────────
#
# Receives:  TextFrame     (AI's complete response text)
# Emits:     AIStatusFrame (ai_speaking=True  — BEFORE audio)
#            AIAudioFrame  (WAV audio bytes)
#            AIStatusFrame (ai_speaking=False — AFTER audio)
#

class SarvamTTSService(FrameProcessor):
    """
    Calls the Sarvam TTS API to synthesize speech from text.

    Handles Sarvam's ~450-character limit by truncating at the last
    sentence boundary within the limit.
    """

    SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
    TTS_CHAR_LIMIT  = 450

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._http    = httpx.AsyncClient(timeout=30.0)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            await self._synthesize(frame.text)
        else:
            await self.push_frame(frame, direction)

    def _truncate(self, text: str) -> str:
        """
        Keep text within Sarvam's per-item character limit.
        Prefer cutting at a sentence boundary to avoid awkward cut-offs.
        """
        if len(text) <= self.TTS_CHAR_LIMIT:
            return text
        truncated = text[:self.TTS_CHAR_LIMIT]
        for punct in (".", "?", "!"):
            last = truncated.rfind(punct)
            if last > self.TTS_CHAR_LIMIT // 2:
                return truncated[:last + 1]
        return truncated

    async def _synthesize(self, text: str):
        try:
            tts_text = self._truncate(text)
            logger.info(f"TTS: synthesizing ({len(tts_text)} chars): {tts_text[:60]!r}…")

            resp = await self._http.post(
                self.SARVAM_TTS_URL,
                headers={
                    "api-subscription-key": self._api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": [tts_text],
                    "target_language_code": "en-IN",
                    "speaker": "anushka",
                    "model": "bulbul:v2",
                },
            )

            if resp.status_code != 200:
                logger.error(f"TTS HTTP {resp.status_code}: {resp.text[:200]}")
                return

            audio_b64 = resp.json().get("audios", [""])[0]
            if not audio_b64:
                logger.error("TTS: empty audio in response")
                return

            wav_bytes = base64.b64decode(audio_b64)
            logger.info(f"TTS: got {len(wav_bytes)} bytes of audio")

            # Signal AI started → send actual audio → signal AI stopped
            await self.push_frame(AIStatusFrame(ai_speaking=True))
            await self.push_frame(AIAudioFrame(audio_bytes=wav_bytes))
            await self.push_frame(AIStatusFrame(ai_speaking=False))

        except Exception as e:
            logger.error(f"TTS error: {e}")
            # Make sure we always reset the speaking flag even on error
            await self.push_frame(AIStatusFrame(ai_speaking=False))

    async def cleanup(self):
        await self._http.aclose()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  OutputSink
# ─────────────────────────────────────────────────────────────────────────────
#
# The LAST processor in the pipeline. Instead of pushing frames further
# downstream (there's nowhere left to go), it puts them onto an asyncio.Queue.
# main.py reads from that queue and sends the data back to the browser.
#

class OutputSink(FrameProcessor):
    """
    Collects pipeline output frames and puts them on an asyncio.Queue.
    The WebSocket handler in main.py reads from this queue to decide
    what to send back to the browser.

    Handles:
      - TranscriptDisplayFrame → browser chat display
      - AIAudioFrame           → browser audio playback
      - AIStatusFrame          → speaking indicator in the browser UI
      - EndFrame               → tells main.py the pipeline is shutting down
    """

    def __init__(self, output_queue: asyncio.Queue, **kwargs):
        super().__init__(**kwargs)
        self._q = output_queue

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Put our custom output frames onto the queue for main.py to read
        if isinstance(frame, (
            TranscriptDisplayFrame,
            AIAudioFrame,
            AIStatusFrame,
            EndFrame,
        )):
            await self._q.put(frame)

        # ALWAYS push every frame downstream, even ones we don't handle.
        #
        # WHY THIS IS CRITICAL:
        # Pipecat sends system frames (StartFrame, StopFrame, HeartbeatFrame,
        # CancelFrame, etc.) through the whole pipeline for lifecycle management.
        # StartFrame in particular MUST reach Pipeline::Sink or the pipeline
        # stays stuck in "Starting. Waiting for StartFrame..." forever and
        # never processes any audio.
        #
        # Without this line: pipeline hangs, no speech is ever processed.
        # With this line:    system frames flow through and pipeline starts.
        await self.push_frame(frame, direction)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  VoicePipelineManager
# ─────────────────────────────────────────────────────────────────────────────
#
# One instance per WebSocket connection.
#
# Responsibilities:
#   - Assemble and start the Pipecat pipeline.
#   - Provide push_audio() so main.py can inject browser audio.
#   - Expose output_queue so main.py can read pipeline results.
#   - Provide interrupt() and stop() for lifecycle management.
#

class VoicePipelineManager:
    """
    Manages the Pipecat pipeline for a single WebSocket connection.

    Usage in main.py:
        manager = VoicePipelineManager()
        await manager.start()

        # Feed audio in:
        await manager.push_audio(pcm_bytes, sample_rate)

        # Read output:
        frame = await manager.output_queue.get()

        # Shut down:
        await manager.stop()
    """

    def __init__(self):
        # Queue that OutputSink fills; main.py reads from this
        self.output_queue: asyncio.Queue = asyncio.Queue()

        # ── Per-session thread_id (Phase 8) ────────────────────────────────
        # This UUID uniquely identifies this WebSocket connection's conversation.
        # LangGraph's MemorySaver uses it as the key to store/retrieve history.
        # Each new connection → new UUID → fresh isolated conversation memory.
        self.thread_id: str = str(uuid.uuid4())
        logger.info(f"VoicePipelineManager: new session thread_id={self.thread_id[:8]}…")

        # ── Build processors ───────────────────────────────────────────────
        # Each is a separate object — they can be individually configured,
        # replaced, or inspected without touching any other stage.
        self._vad  = VADProcessor()           # VAD starts at default 48 kHz; updated on init msg
        self._stt  = SarvamSTTService(api_key=SARVAM_API_KEY)
        # Phase 8: GroqLangGraphProcessor replaces GroqLLMProcessor.
        # The thread_id links this processor to its conversation in MemorySaver.
        self._llm  = GroqLangGraphProcessor(thread_id=self.thread_id)
        self._tts  = SarvamTTSService(api_key=SARVAM_API_KEY)
        self._sink = OutputSink(output_queue=self.output_queue)

        # ── Wire them together in a Pipeline ──────────────────────────────
        # Pipeline([a, b, c]) connects a → b → c.
        # A frame pushed downstream by a arrives at b.process_frame(), and so on.
        self._pipeline = Pipeline([
            self._vad,
            self._stt,
            self._llm,
            self._tts,
            self._sink,
        ])

        # ── Create PipelineTask ────────────────────────────────────────────
        # allow_interruptions=True: if a new utterance arrives while a previous
        # one is still being processed, Pipecat can interrupt the current work
        # and start fresh. Enables natural conversation flow.
        # In pipecat >= 0.0.100, `params` is keyword-only.
        # enable_rtvi=False: pipecat 0.0.108 injects an RTVIProcessor by default
        # (visible in logs as "PipelineTask::Source -> RTVIProcessor -> Pipeline").
        # RTVI is a protocol for Daily.co and similar transports — we're not using
        # it. Leaving it enabled adds an unnecessary layer that can interfere with
        # raw AudioRawFrame injection via task.queue_frame().
        self._task   = PipelineTask(
            self._pipeline,
            params=PipelineParams(allow_interruptions=True),
            enable_rtvi=False,
        )
        self._runner      = PipelineRunner()
        self._runner_coro = None    # asyncio Task for the runner coroutine

    def update_sample_rate(self, rate: int):
        """Call this once the browser sends its init metadata with the real sample rate."""
        self._vad.update_sample_rate(rate)

    async def start(self):
        """
        Launch the pipeline runner as a background asyncio Task.
        The pipeline will now wait for frames to be queued into it.
        """
        # PipelineRunner.run() is a coroutine that blocks until the pipeline ends.
        # We wrap it in create_task() so it runs concurrently with the WebSocket loop.
        self._runner_coro = asyncio.create_task(
            self._runner.run(self._task),
            name="pipecat-pipeline-runner",
        )
        logger.info("VoicePipelineManager: pipeline started")

    async def push_audio(self, pcm_bytes: bytes, sample_rate: int = 48000):
        """
        Inject a raw PCM audio chunk from the browser into the pipeline.
        VADProcessor will accumulate these until a full utterance is detected.

        pcm_bytes   – 16-bit little-endian mono PCM (as received from browser)
        sample_rate – native rate of the browser (typically 48000 Hz)
        """
        frame = AudioRawFrame(
            audio=pcm_bytes,
            sample_rate=sample_rate,
            num_channels=1,
        )
        # queue_frame() injects the frame at the start of the pipeline
        await self._task.queue_frame(frame)

    async def interrupt(self):
        """
        Cancel the current pipeline processing (e.g., user interrupted AI speech).
        The pipeline will still be alive; new audio can be pushed immediately after.
        """
        await self._task.cancel()
        logger.info("VoicePipelineManager: pipeline interrupted")

    async def stop(self):
        """
        Gracefully shut down the pipeline when the WebSocket disconnects.
        Sends EndFrame which flows through to OutputSink, where it signals
        the send_loop in main.py to exit.
        """
        try:
            await self._task.queue_frame(EndFrame())
            if self._runner_coro and not self._runner_coro.done():
                await asyncio.wait_for(self._runner_coro, timeout=3.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        logger.info("VoicePipelineManager: pipeline stopped")

    def clear_memory(self):
        """
        Reset conversation memory by assigning a new thread_id.
        The LangGraph MemorySaver will see a fresh thread and start clean.
        """
        self._llm.reset_thread()
        self.thread_id = self._llm._thread_id
        logger.info("VoicePipelineManager: memory cleared (new thread_id)")
