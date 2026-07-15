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

FRAME FLOW (Phase 7+)
──────────────────────────────────────────────────────────────
  WebSocket binary
        │  AudioRawFrame (raw 16-bit PCM from browser)
        ▼
  ┌─────────────────┐
  │  VADProcessor   │  Voice Activity Detection.
  │                 │  Buffers audio, detects speech start/end,
  │                 │  emits SpeechEndFrame and BargeInDetectedFrame.
  └────────┬────────┘
           │  SpeechEndFrame / BargeInDetectedFrame
           ▼
  ┌──────────────────────┐
  │  SarvamSTTService    │  Calls Sarvam ASR API.
  │                      │  Emits TranscriptionFrame, TranscriptDisplayFrame,
  │                      │  LanguageDetectedFrame, EmotionHintFrame.
  └──────────┬───────────┘
             │  TranscriptionFrame + side-channel frames
             ▼
  ┌──────────────────────┐
  │ GroqLangGraphProcessor│ Streaming LLM with LangGraph memory.
  │                      │  Emits AIThinkingFrame, TextFrame per sentence,
  │                      │  TranscriptDisplayFrame.
  └──────────┬───────────┘
             │  TextFrame (one per sentence)
             ▼
  ┌──────────────────────┐
  │  SarvamTTSService    │  Calls Sarvam TTS API per sentence.
  │                      │  Handles LanguageDetectedFrame for auto-switch.
  │                      │  Emits AIAudioFrame per sentence.
  └──────────┬───────────┘
             │  AIAudioFrame (WAV bytes per sentence)
             ▼
  ┌──────────────────────┐
  │  OutputSink          │  Puts frames onto asyncio.Queue.
  └──────────────────────┘
"""

import array
import asyncio
import base64
import io
import json
import os
import re
import tempfile
import time
import torch
import uuid
import wave
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urlencode

import httpx
import websockets
from loguru import logger

# ─────────────────────────────────────────────────────────────────────────────
# Pipecat Core Imports
# ─────────────────────────────────────────────────────────────────────────────
from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

from app.config import (
    SARVAM_API_KEY,
    STT_STREAMING,
    TTS_STREAMING,
    SARVAM_STT_MODEL,
    SARVAM_STT_MODE,
    SARVAM_TTS_MODEL,
    SARVAM_TTS_SPEAKER,
)
from app.num_to_words import spell_digits as _spell_digits

# ─────────────────────────────────────────────────────────────────────────────
# Load Silero VAD model once at module level (not per-connection)
# ─────────────────────────────────────────────────────────────────────────────
logger.info("VAD: loading Silero VAD model…")
_silero_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    verbose=False,
    trust_repo=True,
)
_silero_model.eval()
logger.info("VAD: Silero VAD model ready")


# ─────────────────────────────────────────────────────────────────────────────
# Custom Frame Types
# ─────────────────────────────────────────────────────────────────────────────

class SpeechEndFrame(Frame):
    """
    Emitted by VADProcessor when a full utterance has been detected.
    Carries the complete buffered WAV audio (already resampled to 16 kHz).
    """
    def __init__(self, audio_bytes: bytes, sample_rate: int = 16000):
        super().__init__()
        self.audio_bytes = audio_bytes
        self.sample_rate = sample_rate


class SpeechStartedFrame(Frame):
    """
    Streaming STT: emitted by VADProcessor exactly once per utterance, the
    moment speech is CONFIRMED active (the same transition that today only
    flips the internal _is_speech_active flag — see MIN_SPEECH_CHUNKS /
    MIN_SPEECH_CHUNKS_BARGEIN below). Lets SarvamSTTStreamingService open its
    per-utterance WebSocket to Sarvam as early as possible, so the connection
    handshake overlaps with the user still talking instead of landing on the
    latency-critical tail after they stop.

    Internal frame only — never forwarded to OutputSink/main.py/the browser.
    """
    pass


class SpeechChunkFrame(Frame):
    """
    Streaming STT: emitted by VADProcessor for every 16 kHz PCM chunk added to
    _audio_buffer while an utterance is active (mirrors exactly what the
    batch WAV would contain — both confirmed-speech chunks and the trailing
    silence chunks before the utterance is declared over). Lets
    SarvamSTTStreamingService stream audio to Sarvam in near-real-time instead
    of waiting for SpeechEndFrame's complete WAV.

    Internal frame only — never forwarded to OutputSink/main.py/the browser.
    """
    def __init__(self, pcm_16k: bytes):
        super().__init__()
        self.pcm_16k = pcm_16k


class TranscriptDisplayFrame(Frame):
    """
    Emitted by STT (for user text) and LLM (for AI text).
    Carries a line of conversation text that the browser chat UI should display.
    """
    def __init__(self, text: str, speaker: str = "user"):
        super().__init__()
        self.text = text
        self.speaker = speaker


class AIAudioFrame(Frame):
    """
    Emitted by SarvamTTSService.
    Carries synthesized WAV audio bytes for one sentence.
    """
    def __init__(self, audio_bytes: bytes):
        super().__init__()
        self.audio_bytes = audio_bytes


class AIStatusFrame(Frame):
    """
    Emitted by SarvamTTSService before and after generating audio.
    Lets main.py track whether the AI is currently speaking.
    """
    def __init__(self, ai_speaking: bool):
        super().__init__()
        self.ai_speaking = ai_speaking


# ── NEW FRAMES (Features 4, 6, 7, 10) ──────────────────────────────────────

class AIThinkingFrame(Frame):
    """
    Feature 6: Typing Indicator.
    Emitted by GroqLangGraphProcessor before the first LLM sentence is ready
    (thinking=True) and removed once the first sentence is sent to TTS
    (thinking=False). Browser shows animated "AI is thinking…" dots.
    """
    def __init__(self, thinking: bool):
        super().__init__()
        self.thinking = thinking


class LanguageDetectedFrame(Frame):
    """
    Feature 7: Language Auto-Switch.
    Emitted by SarvamSTTService after reading the language_code field from
    Sarvam ASR response. Flows downstream to SarvamTTSService which switches
    its target_language_code accordingly. Also forwarded to browser for badge.
    """
    def __init__(self, language_code: str):
        super().__init__()
        self.language_code = language_code


class EmotionHintFrame(Frame):
    """
    Feature 10: Emotion/Tone Detection.
    Emitted by VADProcessor (energy-based) or SarvamSTTService (confidence-based).
    Consumed by GroqLangGraphProcessor to adjust the LLM system prompt.
    hint: "neutral" | "hesitant" | "agitated"
    """
    def __init__(self, hint: str):
        super().__init__()
        self.hint = hint  # "neutral", "hesitant", "agitated"


class BargeInDetectedFrame(Frame):
    """
    Feature 4: Barge-in Detection.
    Emitted by VADProcessor when speech is detected while AI is speaking
    (barge_in_mode=True). main.py receives this from OutputSink and
    immediately interrupts the pipeline + stops browser audio.
    """
    pass


class NoSpeechDetectedFrame(Frame):
    """
    Emitted by SarvamSTTService when an utterance transcribes to nothing
    usable (empty, too short, or a bare filler word — see FILLER_WORDS).
    Normally this is just noise being filtered. But if it followed a
    barge-in, it means the "interruption" wasn't real speech: consumed by
    GroqLangGraphProcessor to RESUME any AI turn that was paused for that
    barge-in instead of silently abandoning it. A no-op if no turn is paused.
    """
    pass


class LLMTurnDoneFrame(Frame):
    """
    Emitted by GroqLangGraphProcessor after all TextFrames for a turn have
    been pushed. Travels through the pipeline IN ORDER so SarvamTTSService
    receives it only after every TextFrame for this turn has been processed.
    This replaces the external flush() call which had a race condition:
    flush() could be called before the last TextFrame reached process_frame.
    """
    pass


class EndCallFrame(Frame):
    """
    Agent-initiated hangup. Emitted by GroqLangGraphProcessor AFTER the goodbye
    turn's audio has been delivered, when stream_agent reported _meta_out
    ["end_call"]=True (the LLM appended [END_CALL] to its final message).

    main.py's send_loop receives this from OutputSink and closes the WebSocket
    — but only after the goodbye audio has been flushed to the browser, so the
    customer always hears the farewell before the line drops.
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 1.  VADProcessor  (Voice Activity Detection — Silero VAD)
# ─────────────────────────────────────────────────────────────────────────────
#
# NEW in this version:
#   Feature 4 (Barge-in):  barge_in_mode lowers silence threshold so the VAD
#                           responds faster, and emits BargeInDetectedFrame the
#                           moment speech starts while AI is playing audio.
#   Feature 10 (Emotion):  tracks average speech energy per utterance. If the
#                           energy is > 2× baseline → emit EmotionHintFrame("agitated").
#
# Receives:  AudioRawFrame  (one ~85ms PCM chunk from the browser)
# Emits:     SpeechEndFrame         (one complete utterance as WAV)
#            BargeInDetectedFrame   (when barge-in speech starts)
#            EmotionHintFrame       (when high energy detected)
#

class VADProcessor(FrameProcessor):
    """
    Voice Activity Detection using Silero VAD neural network.

    Algorithm:
      1. Resample each incoming PCM chunk from browser rate → 16 kHz.
      2. Run the chunk through Silero VAD → speech probability (0.0–1.0).
      3. If prob > SPEECH_THRESHOLD for MIN_SPEECH_CHUNKS consecutive
         chunks → speech started, begin buffering.
      4. Once speech started, if prob < SPEECH_THRESHOLD for
         silence_needed consecutive chunks → utterance complete.
      5. Hard-cap at MAX_BUFFER_CHUNKS to avoid infinite wait.
    """

    # ── Silero thresholds ─────────────────────────────────────────────────────
    # Each browser frame ≈ 85 ms (4096 samples @ ~48 kHz), so chunk counts below
    # translate roughly: 1 chunk ≈ 85 ms, 2 ≈ 170 ms, 3 ≈ 255 ms.
    SPEECH_THRESHOLD     = 0.5   # Silero probability above this = speech
    MIN_SPEECH_CHUNKS    = 3     # ~0.25 s of confirmed speech before buffering (normal turns)
    # Barge-in must feel like real conversation: the instant the user speaks over
    # the AI, stop. Fire after just 2 confirmed speech chunks (~170 ms) instead of
    # 3 — fast enough to feel instant, still enough to reject a single-chunk noise
    # blip. Normal (non-barge-in) detection keeps 3 to avoid false starts on the
    # user's own turns.
    MIN_SPEECH_CHUNKS_BARGEIN = 2
    # End-of-utterance silence window. 0.42 s was too aggressive — a natural
    # mid-sentence "thinking" pause (e.g. "…but I want to … switch") exceeded it,
    # so the agent cut the customer off. 12 chunks ≈ 1.0 s lets people pause to
    # think without being interrupted, while still feeling responsive once they
    # actually stop. Barge-in stays snappy (it only ends an already-detected
    # interruption, where the user is clearly committed to speaking).
    SILENCE_CHUNKS_NEEDED = 12  # ~1.0 s of quiet = utterance ended (normal mode)
    SILENCE_CHUNKS_BARGEIN = 4  # Feature 4: faster end-of-speech during barge-in
    MAX_BUFFER_CHUNKS    = 180  # ~15 s safety cap (longer sentences now allowed)

    # ── Noise-rejection energy gate ───────────────────────────────────────────
    # Silero's probability alone occasionally scores broadband ambient noise
    # (fans, distant traffic/chatter, hums) at or above SPEECH_THRESHOLD. A
    # chunk must ALSO be meaningfully louder than the learned ambient noise
    # floor (_noise_floor_energy) to count as speech. Starting points — tune
    # against real call logs (VAD: rejected noise-like chunk / VAD: BARGE-IN
    # fired debug/info lines) if false triggers persist or real speech gets
    # rejected.
    NOISE_FLOOR_ALPHA      = 0.05   # slow EMA — one loud moment shouldn't raise the floor
    ENERGY_GATE_MULTIPLIER = 1.6    # a chunk must be this many× louder than the floor
    MIN_ABSOLUTE_ENERGY    = 150.0  # floor for a near-silent room where noise_floor≈0

    # ── Sample rate constants ─────────────────────────────────────────────────
    TARGET_SAMPLE_RATE   = 16000  # Silero and Sarvam ASR both expect 16 kHz

    def __init__(self, browser_sample_rate: int = 48000, **kwargs):
        super().__init__(**kwargs)
        self._browser_rate = browser_sample_rate
        self._model = _silero_model
        self._barge_in_mode = False       # Feature 4: set True when AI is speaking
        self._barge_in_signaled = False   # Feature 4: prevent multiple barge-in signals
        # Feature 10: energy tracking for emotion detection
        self._energy_sum = 0.0
        self._energy_count = 0
        self._energy_baseline = 0.0   # rolling average of normal speech energy
        # Ambient noise-floor estimate for the energy gate — persists across
        # utterances for the life of the connection (NOT reset in
        # _reset_vad_state, unlike the per-utterance buffers below).
        self._noise_floor_energy = 0.0
        self._reset_vad_state()

    def update_sample_rate(self, rate: int):
        """Called by main.py once the browser sends its init metadata."""
        self._browser_rate = rate
        logger.info(f"VAD: browser sample rate updated to {rate} Hz")

    def set_barge_in_mode(self, enabled: bool):
        """
        Feature 4: Enable/disable barge-in mode.
        In barge-in mode:
          - Silence threshold is lowered (SILENCE_CHUNKS_BARGEIN = 3 chunks)
            so the VAD ends the utterance faster (faster response).
          - When speech first starts, BargeInDetectedFrame is emitted so
            main.py can immediately interrupt the AI.
        Call with True when AI starts speaking, False when AI stops.
        """
        self._barge_in_mode = enabled
        self._barge_in_signaled = False  # reset signal for new AI turn
        logger.debug(f"VAD: barge_in_mode={'ON' if enabled else 'OFF'}")

    def _reset_vad_state(self):
        """Clear all VAD buffers and reset Silero hidden state after each utterance."""
        self._audio_buffer: List[bytes] = []
        self._speech_chunks_seen  = 0
        self._silence_chunk_count = 0
        self._is_speech_active    = False
        self._silero_leftover: list = []
        self._energy_sum = 0.0
        self._energy_count = 0
        self._model.reset_states()

    # ── Resampling ────────────────────────────────────────────────────────────

    def _resample(self, pcm_bytes: bytes) -> bytes:
        """Downsample from browser rate to 16 kHz.

        Averages each source block instead of picking a single nearest sample
        (naive decimation). Decimation left frequency content above the new
        Nyquist rate (8 kHz) unfiltered, which aliases down into the audible
        band and degrades the signal Silero sees — contributing to spurious
        speech-probability spikes on non-speech audio. Block-averaging is a
        cheap box-car low-pass that meaningfully reduces that aliasing without
        pulling in a DSP dependency (e.g. scipy) just for this.
        """
        if self._browser_rate == self.TARGET_SAMPLE_RATE:
            return pcm_bytes
        ratio = self._browser_rate / self.TARGET_SAMPLE_RATE
        samples = array.array('h', pcm_bytes[:len(pcm_bytes) & ~1])
        out_len = int(len(samples) / ratio)
        out = array.array('h')
        for i in range(out_len):
            start = int(i * ratio)
            end   = min(int((i + 1) * ratio), len(samples))
            if end <= start:
                end = start + 1
            block = samples[start:end]
            out.append(sum(block) // len(block) if block else 0)
        return out.tobytes()

    # ── Silero inference ──────────────────────────────────────────────────────

    SILERO_WINDOW = 512

    def _speech_prob(self, pcm_16k: bytes) -> float:
        """Run one 16-kHz PCM chunk through Silero VAD. Returns max speech prob."""
        samples = array.array('h', pcm_16k[:len(pcm_16k) & ~1])
        if not samples:
            return 0.0
        combined = self._silero_leftover + list(samples)
        self._silero_leftover = []
        probs = []
        i = 0
        while i + self.SILERO_WINDOW <= len(combined):
            window = combined[i: i + self.SILERO_WINDOW]
            float_win = [s / 32768.0 for s in window]
            tensor = torch.tensor(float_win, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                probs.append(self._model(tensor, self.TARGET_SAMPLE_RATE).item())
            i += self.SILERO_WINDOW
        self._silero_leftover = combined[i:]
        return max(probs) if probs else 0.0

    # ── WAV packing ──────────────────────────────────────────────────────────

    def _pcm_to_wav(self, pcm_bytes: bytes) -> bytes:
        """Wrap raw 16-kHz PCM bytes in a WAV header for Sarvam ASR."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.TARGET_SAMPLE_RATE)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    # ── Feature 10: energy helper ────────────────────────────────────────────

    def _track_energy(self, pcm_16k: bytes):
        """Track RMS energy of speech chunks for emotion detection."""
        samples = array.array('h', pcm_16k[:len(pcm_16k) & ~1])
        if samples:
            energy = sum(abs(s) for s in samples) / len(samples)
            self._energy_sum += energy
            self._energy_count += 1

    # ── Pipecat frame processing ──────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame):
            await self._process_audio_chunk(frame.audio)
        else:
            await self.push_frame(frame, direction)

    def _resample_and_score(self, raw_pcm: bytes) -> tuple[bytes, float]:
        """Resample to 16 kHz and run Silero inference. Both are CPU-bound, so
        they're done together in ONE executor call (see _process_audio_chunk)
        — previously only the Silero call was offloaded, leaving resampling on
        the event loop."""
        pcm_16k = self._resample(raw_pcm)
        prob = self._speech_prob(pcm_16k)
        return pcm_16k, prob

    def _chunk_energy(self, pcm_16k: bytes) -> float:
        """Mean absolute amplitude of a 16-kHz PCM chunk — a cheap loudness proxy
        used by the noise-rejection energy gate below."""
        samples = array.array('h', pcm_16k[:len(pcm_16k) & ~1])
        return sum(abs(s) for s in samples) / len(samples) if samples else 0.0

    async def _process_audio_chunk(self, raw_pcm: bytes):
        """Run one chunk of browser audio through the Silero VAD state machine."""
        # Silero inference (and now resampling too) is CPU-bound torch/array
        # work (~12×/sec). Running it inline blocked the event loop, adding
        # jitter to audio delivery and slowing barge-in during playback
        # (Bug #10). Offload both to the default thread pool so the loop stays
        # free for WebSocket/HTTP I/O. Safe because chunks are processed
        # strictly one at a time (process_frame awaits each in order), so the
        # _silero_leftover state carried across calls is never raced.
        loop = asyncio.get_running_loop()
        pcm_16k, prob = await loop.run_in_executor(None, self._resample_and_score, raw_pcm)
        energy = self._chunk_energy(pcm_16k)

        # ── Adaptive noise-floor calibration ──────────────────────────────────
        # Continuously learn the room's ambient noise level from chunks Silero
        # itself scores as non-speech, so the energy gate below adapts to a
        # quiet room vs. a noisy office instead of using one fixed global
        # amplitude threshold. Slow EMA (persists across utterances — NOT
        # reset in _reset_vad_state) so one loud moment doesn't quickly raise
        # the floor and mask genuine speech right after it.
        if prob < self.SPEECH_THRESHOLD:
            self._noise_floor_energy = (
                energy if self._noise_floor_energy == 0.0
                else (1 - self.NOISE_FLOOR_ALPHA) * self._noise_floor_energy
                     + self.NOISE_FLOOR_ALPHA * energy
            )

        # A chunk counts as speech only if Silero's probability AND a loudness
        # sanity check both agree. Silero alone occasionally scores broadband
        # background noise (fans, distant traffic/chatter, hums) at or above
        # SPEECH_THRESHOLD; requiring it to also be meaningfully louder than
        # the learned ambient floor rejects those without a fixed,
        # environment-specific absolute threshold. Skipped once speech is
        # already confirmed active — natural mid-utterance dips in volume
        # (quiet syllables, breaths) shouldn't be second-guessed.
        energy_gate = max(self.MIN_ABSOLUTE_ENERGY, self._noise_floor_energy * self.ENERGY_GATE_MULTIPLIER)
        is_speech = prob >= self.SPEECH_THRESHOLD and (self._is_speech_active or energy >= energy_gate)
        if prob >= self.SPEECH_THRESHOLD and not is_speech:
            logger.debug(
                f"VAD: rejected noise-like chunk (prob={prob:.2f} energy={energy:.0f} "
                f"gate={energy_gate:.0f}) — Silero-positive but too quiet vs. ambient floor"
            )

        if is_speech:
            # ── SPEECH chunk ─────────────────────────────────────────────────
            self._silence_chunk_count = 0
            self._audio_buffer.append(pcm_16k)
            self._speech_chunks_seen += 1

            # Feature 4: BARGE-IN — fire as soon as the user speaks OVER the AI,
            # at a lower threshold (~170 ms) than normal utterance detection, so
            # interruption feels like real conversation. Decoupled from the
            # _is_speech_active gate below (which needs more chunks) — waiting for
            # that made barge-in sluggish.
            if (self._barge_in_mode and not self._barge_in_signaled
                    and self._speech_chunks_seen >= self.MIN_SPEECH_CHUNKS_BARGEIN):
                self._barge_in_signaled = True
                logger.info(f"VAD: BARGE-IN fired after {self._speech_chunks_seen} chunks (prob={prob:.2f})")
                await self.push_frame(BargeInDetectedFrame())

            just_activated = False
            if not self._is_speech_active:
                if self._speech_chunks_seen >= self.MIN_SPEECH_CHUNKS:
                    self._is_speech_active = True
                    just_activated = True
                    logger.debug(f"VAD: speech STARTED (prob={prob:.2f}, barge_in={self._barge_in_mode})")

            # Feature 10: track energy during active speech
            if self._is_speech_active:
                self._track_energy(pcm_16k)

                # Streaming STT: on the activation transition, backfill every
                # chunk that accumulated in _audio_buffer during the
                # MIN_SPEECH_CHUNKS/MIN_SPEECH_CHUNKS_BARGEIN confirmation
                # window (appended above, before _is_speech_active was true,
                # so never individually streamed yet) so SarvamSTTStreamingService
                # sees the exact same audio the batch WAV would contain — no
                # gap at the start of the utterance. Then stream this call's
                # own chunk exactly once, same as every subsequent call.
                if just_activated:
                    await self.push_frame(SpeechStartedFrame())
                    for buffered_chunk in self._audio_buffer[:-1]:
                        await self.push_frame(SpeechChunkFrame(buffered_chunk))
                await self.push_frame(SpeechChunkFrame(pcm_16k))

        else:
            # ── SILENCE chunk (or a Silero-positive chunk that failed the
            # loudness gate — treated the same as silence for state purposes) ──
            if self._is_speech_active:
                self._silence_chunk_count += 1
                self._audio_buffer.append(pcm_16k)
                # Streaming STT: trailing silence chunks are part of the same
                # utterance's audio (the batch WAV includes them too) — stream
                # them same as active-speech chunks above.
                await self.push_frame(SpeechChunkFrame(pcm_16k))

                # Feature 4: use shorter silence window during barge-in for faster response
                silence_needed = (
                    self.SILENCE_CHUNKS_BARGEIN if self._barge_in_mode
                    else self.SILENCE_CHUNKS_NEEDED
                )
                silence_ended = self._silence_chunk_count >= silence_needed
                hard_cap_hit  = len(self._audio_buffer) >= self.MAX_BUFFER_CHUNKS

                if silence_ended or hard_cap_hit:
                    reason = "silence" if silence_ended else "hard-cap"
                    logger.info(f"VAD: utterance END ({reason}, last_prob={prob:.2f})")
                    await self._emit_utterance()
            else:
                # Not yet confirmed speech — a silence/rejected chunk here means
                # the prior above-threshold chunk(s) were an isolated blip, not
                # the start of a real utterance. Reset the pre-activation
                # counter so only TRULY CONSECUTIVE qualifying chunks can cross
                # MIN_SPEECH_CHUNKS / MIN_SPEECH_CHUNKS_BARGEIN.
                #
                # ROOT CAUSE this fixes: previously this branch did nothing, so
                # _speech_chunks_seen was NEVER reset before activation. Sparse
                # noise blips (AC cycling, clicks, distant chatter) scattered
                # across many seconds — with real silence in between — could
                # silently accumulate past the threshold and falsely trigger an
                # utterance or a barge-in, even though no chunk was ever part of
                # a real, continuous utterance.
                self._speech_chunks_seen = 0

    async def _emit_utterance(self):
        """Package buffered audio into WAV and push a SpeechEndFrame downstream."""
        if not self._audio_buffer:
            self._reset_vad_state()
            return

        # Feature 10: check if user was speaking unusually loudly (emotion = agitated)
        if self._energy_count > 0:
            avg_energy = self._energy_sum / self._energy_count
            if self._energy_baseline == 0.0:
                self._energy_baseline = avg_energy   # first utterance sets baseline
            elif avg_energy > self._energy_baseline * 2.0:
                logger.info(f"VAD: high energy detected ({avg_energy:.0f} vs baseline {self._energy_baseline:.0f}) → agitated")
                await self.push_frame(EmotionHintFrame(hint="agitated"))
            # Update rolling baseline (70% old, 30% new)
            self._energy_baseline = 0.7 * self._energy_baseline + 0.3 * avg_energy

        raw_pcm   = b"".join(self._audio_buffer)
        wav_bytes = self._pcm_to_wav(raw_pcm)

        logger.info(f"VAD: emitting SpeechEndFrame — {len(raw_pcm)} bytes PCM → {len(wav_bytes)} bytes WAV")
        await self.push_frame(SpeechEndFrame(audio_bytes=wav_bytes, sample_rate=self.TARGET_SAMPLE_RATE))

        self._reset_vad_state()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SarvamSTTService  (Speech-to-Text)
# ─────────────────────────────────────────────────────────────────────────────
#
# NEW in this version:
#   Feature 7 (Language): reads language_code from Sarvam response and emits
#                         LanguageDetectedFrame. Tracks detected language for
#                         subsequent STT requests (auto-adapts over turns).
#   Feature 10 (Emotion): reads confidence from response. Low confidence
#                         (< 0.6) → emit EmotionHintFrame("hesitant").
#
# Receives:  SpeechEndFrame  (WAV audio of one utterance)
# Emits:     TranscriptionFrame      (text → LLM)
#            TranscriptDisplayFrame  (text → browser chat UI)
#            LanguageDetectedFrame   (detected language → TTS + browser badge)
#            EmotionHintFrame        (low confidence → LLM prompt adjustment)
#

class SarvamSTTService(FrameProcessor):
    """
    Calls the Sarvam ASR API to transcribe one complete utterance.
    Also detects language and transcription confidence for downstream features.
    """

    SARVAM_ASR_URL = "https://api.sarvam.ai/speech-to-text"

    FILLER_WORDS = {
        "yes", "no", "ok", "okay", "hmm", "uh", "um", "ah", "oh",
        "huh", "hm", "yeah", "yep", "nope", "hey"
    }

    # Feature 7: mapping from Sarvam short codes to BCP-47 language tags
    LANG_NORMALIZE = {
        "hi": "hi-IN", "ta": "ta-IN", "te": "te-IN", "kn": "kn-IN",
        "en": "en-IN", "mr": "mr-IN", "bn": "bn-IN", "gu": "gu-IN",
        "pa": "pa-IN", "ml": "ml-IN", "or": "or-IN",
    }

    # Romanized Indian-language word lists keyed by BCP-47 code.
    # Used to recover the correct language when Sarvam mis-labels romanized
    # Indian speech as "en" (a known limitation of auto-detect mode).
    # Words chosen as high-frequency function words that cannot plausibly appear
    # in real English sentences.
    ROMANIZED_MARKERS: dict[str, set] = {
        "hi-IN": {
            "kya", "aap", "main", "mein", "hai", "hain", "nahi", "nahin",
            "baat", "saath", "mere", "mera", "meri", "tum", "tumhara",
            "sakte", "sakta", "sakti", "chahiye", "hoga", "yeh", "woh",
            "kaise", "kahan", "kyun", "kyunki", "lekin", "aur", "agar",
            "toh", "phir", "abhi", "bahut", "thoda", "kuch", "koi",
            "accha", "theek", "haan", "bolo", "batao", "namaste",
        },
        "mr-IN": {
            "majha", "majhi", "mala", "tula", "aahe", "aahes", "naav",
            "kay", "kasa", "kashi", "kashala", "tumhi", "aami", "tyala",
            "tila", "aplya", "ata", "aani", "pan", "jar", "tar", "mhanje",
            "sangto", "sangta", "bagh", "bagha", "yeto", "yete", "jato",
            "jate", "ghara", "shala", "pudhe", "mage", "khup", "thoda",
        },
        "ta-IN": {
            "enna", "naan", "nee", "avan", "aval", "avanga", "vandhen",
            "pogiren", "sollu", "paarunga", "theriyum", "illai", "aamaa",
            "enakku", "unnakku", "ingey", "angey", "eppo", "eppadi",
            "romba", "konjam", "yaarukku", "enna", "solren",
        },
        "te-IN": {
            "nenu", "meeru", "atanu", "aame", "vaallu", "vachchanu",
            "velthanu", "cheppandi", "chudandi", "telusa", "ledu", "avunu",
            "naaku", "meeku", "ikkada", "akkada", "eppudu", "ela",
            "chala", "konchem", "evaru", "emi", "chestanu",
        },
        "kn-IN": {
            "nanu", "neevu", "avanu", "avalu", "avaru", "bartini",
            "hoguttini", "heli", "nodi", "gotthu", "illa", "howdu",
            "nanage", "nimage", "illi", "alli", "yaavaga", "hege",
            "thumba", "swalpa", "yaaru", "yenu", "maaduttini",
        },
        "pa-IN": {
            "main", "tussi", "oh", "assi", "aaya", "gaya", "karo",
            "dekho", "dassi", "pata", "nahi", "haan", "kiddan",
            "kiven", "kithe", "kyon", "bahut", "thoda", "koi",
            "kuch", "sanu", "tenu", "saade", "twaade",
        },
        "gu-IN": {
            "hoon", "tame", "te", "ame", "aavyo", "gayo", "karo",
            "juo", "khabar", "nathi", "haa", "kem", "kyare",
            "kyaan", "kem", "ghanu", "thodu", "koi", "kuch",
            "mane", "tane", "amane", "tamane",
        },
        "bn-IN": {
            "ami", "tumi", "se", "tara", "eshechi", "gechi", "bolo",
            "dekho", "jano", "na", "haan", "kemon", "kobe", "kothay",
            "keno", "onek", "ektu", "ke", "ki", "amake", "tomake",
        },
    }

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self._api_key     = api_key
        self._http        = httpx.AsyncClient(timeout=30.0)
        # Always "unknown" — Sarvam auto-detects every turn independently.
        # We never persist the detected language here because locking to a
        # language would break switching (e.g. user goes English → Hindi → English).
        self._language    = "unknown"

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, SpeechEndFrame):
            await self._transcribe(frame)
        else:
            await self.push_frame(frame, direction)

    async def _transcribe(self, frame: SpeechEndFrame):
        """
        Upload the WAV audio to Sarvam ASR and emit the transcript.

        Feature 7: After the first transcription, we update self._language
        to whatever Sarvam detected, so subsequent requests are sent with
        the right language_code for better accuracy.

        Feature 10: If Sarvam returns a confidence score < 0.6, we emit
        an EmotionHintFrame("hesitant") so the LLM can be more encouraging.
        """
        stt_t0 = time.monotonic()   # LATENCY: measure the ASR round-trip
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="pipecat_utt_")
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(frame.audio_bytes)

            # Model/mode come from config (.env) so Sarvam STT versions can be
            # migrated without code changes. `mode` is only valid on saaras:*
            # models (saarika ignores/rejects it), so add it conditionally.
            stt_data = {
                "model": SARVAM_STT_MODEL,
                # Always "unknown" — let Sarvam auto-detect every turn.
                # This is the only way to correctly handle mid-conversation
                # language switches (English → Hindi → English → Marathi…).
                "language_code": "unknown",
                "with_disfluencies": "false",
            }
            if SARVAM_STT_MODEL.startswith("saaras"):
                stt_data["mode"] = SARVAM_STT_MODE
            with open(tmp_path, "rb") as f:
                resp = await self._http.post(
                    self.SARVAM_ASR_URL,
                    headers={"api-subscription-key": self._api_key},
                    files={"file": ("audio.wav", f, "audio/wav")},
                    data=stt_data,
                )

            if resp.status_code != 200:
                logger.error(f"STT HTTP {resp.status_code}: {resp.text[:200]}")
                return

            resp_json   = resp.json()
            transcript  = resp_json.get("transcript", "").strip()
            stt_ms = int((time.monotonic() - stt_t0) * 1000)
            logger.info(f"STT transcript: {transcript!r}")
            logger.info(f"LATENCY stt_ms={stt_ms}")

            raw_lang   = resp_json.get("language_code", "en")
            confidence = float(resp_json.get("confidence", 1.0))
            await _finalize_transcript(self.push_frame, transcript, raw_lang, confidence)

        except Exception as e:
            logger.error(f"STT error: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    async def cleanup(self):
        await self._http.aclose()


async def _finalize_transcript(
    push_frame,
    transcript: str,
    raw_lang: str,
    confidence: float | None,
) -> None:
    """
    Shared post-processing for a FINALIZED STT result — used by both
    SarvamSTTService (batch HTTP) and SarvamSTTStreamingService (streaming
    WebSocket, see below) so downstream behavior is byte-identical no matter
    which transport produced the transcript: romanized-language override,
    Feature 10 confidence-based emotion hint, filler/noise filtering, and
    frame emission order (display frame before TranscriptionFrame — the
    latter triggers the long-running LLM call, so the user's chat bubble
    must land first).

    confidence=None (the streaming path's response has no confidence score)
    skips the hesitant-hint emission entirely — a documented gap, not a
    silent one; VAD-energy-based "agitated" detection is unaffected since it
    lives entirely in VADProcessor.
    """
    # ── Feature 7: Language detection ──────────────────────────────────────
    # Sarvam returns the detected language code in the response. We normalise
    # it, then apply a romanized-language correction pass: Sarvam's
    # auto-detect sometimes returns "en" for romanized Indian speech (e.g.
    # Hinglish, romanized Marathi) because the text looks Latin-script. We
    # scan the transcript for known function words of each Indian language
    # and override when we get ≥2 hits.
    detected = SarvamSTTService.LANG_NORMALIZE.get(raw_lang, raw_lang)

    if detected == "en-IN" and transcript:
        words = set(transcript.lower().replace(",", " ").replace(".", " ").split())
        best_lang  = None
        best_count = 0
        for lang_code, markers in SarvamSTTService.ROMANIZED_MARKERS.items():
            hits = len(words & markers)
            if hits > best_count:
                best_count = hits
                best_lang  = lang_code
        if best_count >= 2 and best_lang:
            logger.info(
                f"STT: romanized {best_lang} detected ({best_count} markers) — "
                f"overriding language en-IN → {best_lang}"
            )
            detected = best_lang

    logger.info(f"STT: language={detected!r}")
    await push_frame(LanguageDetectedFrame(language_code=detected))

    # ── Feature 10: Emotion hint from confidence ────────────────────────────
    if confidence is not None and confidence < 0.6:
        logger.info(f"STT: low confidence ({confidence:.2f}) → hesitant emotion hint")
        await push_frame(EmotionHintFrame(hint="hesitant"))

    # ── Noise / filler filter ────────────────────────────────────────────────
    cleaned = transcript.lower().rstrip(".,!? ")
    if not cleaned or len(cleaned) <= 2 or cleaned in SarvamSTTService.FILLER_WORDS:
        logger.debug(f"STT: filtered noise/filler {transcript!r}")
        # This utterance produced nothing usable. If it followed a barge-in,
        # that barge-in was a false positive (background noise, not real
        # speech) — NoSpeechDetectedFrame lets GroqLangGraphProcessor RESUME
        # whatever AI turn got cut off instead of silently leaving the call in
        # dead air. A no-op when no turn is paused (e.g. plain background
        # noise between turns, nothing was interrupted).
        await push_frame(NoSpeechDetectedFrame())
        return

    # ── Emit frames downstream ───────────────────────────────────────────────
    # ORDERING: push user display frame FIRST, then the transcription.
    # TranscriptionFrame triggers a long-running LLM call which blocks the
    # pipeline. Pushing the display frame first ensures the user's text
    # bubble appears in the browser BEFORE the AI response audio.
    await push_frame(TranscriptDisplayFrame(text=transcript, speaker="user"))
    await push_frame(TranscriptionFrame(text=transcript, user_id="user", timestamp=""))


# ─────────────────────────────────────────────────────────────────────────────
# 2b.  SarvamSTTStreamingService  (Speech-to-Text over the streaming WebSocket)
# ─────────────────────────────────────────────────────────────────────────────
#
# Flag-gated alternative to SarvamSTTService (batch HTTP), enabled by
# STT_STREAMING=true. Chosen in VoicePipelineManager; when the flag is off
# this class is never instantiated and behavior is byte-identical to before.
#
# WHY THIS EXISTS: the batch path waits for VADProcessor to buffer the WHOLE
# utterance (~1s of trailing silence) before making a single blocking HTTP
# call to Sarvam ASR — the LLM turn cannot start until that full round-trip
# completes. This service instead streams each 16kHz PCM chunk to Sarvam's
# STT WebSocket AS THE USER SPEAKS (fed by VADProcessor's new
# SpeechStartedFrame/SpeechChunkFrame), so ASR compute overlaps their
# speaking time. When VAD detects silence and emits SpeechEndFrame, we send
# a flush signal and the finalized transcript is typically ready almost
# immediately — most of the work already happened during the pause we used
# to spend waiting idle.
#
# This does NOT stream word-by-word interim text to the LLM: Sarvam's
# streaming response is a FINALIZED transcript per utterance (confirmed
# against the real API's documented protocol), not incremental partial
# tokens, and the LLM still only starts once on the one complete transcript
# — feeding partial fragments into a sales-agent LLM turn would produce
# premature, incoherent replies to half a sentence.
#
# FAILS SAFE: any connect failure, mid-utterance drop, or a flush that
# doesn't produce a transcript within FINALIZE_TIMEOUT_SECS falls back to
# the exact same batch HTTP call SarvamSTTService already makes, using the
# complete WAV VADProcessor buffers into SpeechEndFrame regardless of
# whether streaming is in use. Reliability can never regress below the
# pre-streaming behavior — only latency changes.
#
# Sarvam's streaming response has no confidence score, so the Feature 10
# "hesitant" emotion hint (EmotionHintFrame from low ASR confidence) is only
# ever emitted via the batch fallback path, never on a streaming success —
# see _finalize_transcript's confidence=None handling above. A documented
# gap, not a silent one; VAD-energy-based "agitated" detection is unaffected
# since it lives entirely in VADProcessor.
#
# Receives:  SpeechStartedFrame, SpeechChunkFrame, SpeechEndFrame
# Emits:     same contract as SarvamSTTService — TranscriptDisplayFrame,
#            TranscriptionFrame, LanguageDetectedFrame, EmotionHintFrame,
#            NoSpeechDetectedFrame
#

_STT_END_SENTINEL = object()   # marks the queue item that carries SpeechEndFrame


class SarvamSTTStreamingService(FrameProcessor):
    """Streaming Sarvam STT over a per-utterance WebSocket. See section header."""

    SARVAM_STT_WS = "wss://api.sarvam.ai/speech-to-text/ws"
    # Bounded — never block a turn indefinitely waiting on a flush response.
    # On timeout we fall back to the batch path (see _run_utterance).
    FINALIZE_TIMEOUT_SECS = 3.0

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self._api_key = api_key
        # Batch fallback REUSES SarvamSTTService's own HTTP client + _transcribe
        # logic rather than duplicating it — one implementation, two transports.
        self._batch_fallback = SarvamSTTService(api_key=api_key)
        # Per-utterance queue + worker task. Reassigned fresh on every
        # SpeechStartedFrame; the previous utterance's worker (if any) is left
        # to finish independently via its own captured closure variables, not
        # shared mutable state — mirrors SarvamTTSStreamingService's per-turn
        # isolation pattern (pipecat_pipeline.py, SarvamTTSStreamingService).
        self._queue: asyncio.Queue | None = None
        self._worker_task: asyncio.Task | None = None

    # ── Pipecat frame handler ─────────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, SpeechStartedFrame):
            queue = asyncio.Queue()
            self._queue = queue
            self._worker_task = asyncio.create_task(
                self._run_utterance(queue, time.monotonic())
            )

        elif isinstance(frame, SpeechChunkFrame):
            if self._queue is not None:
                self._queue.put_nowait(frame.pcm_16k)
            # else: no SpeechStartedFrame was ever observed for this utterance
            # (shouldn't happen given VADProcessor's emission order) — dropping
            # is safe since SpeechEndFrame's WAV is a complete, independent copy.

        elif isinstance(frame, SpeechEndFrame):
            # Non-blocking handoff: the per-utterance worker task (already
            # running since SpeechStartedFrame) does the flush + finalize +
            # fallback entirely off this call stack. process_frame returns
            # immediately so VAD is free to start buffering the NEXT utterance
            # right away — never repeat the old TTS bug where a slow network
            # wait inside process_frame stalled the whole pipeline.
            if self._queue is not None:
                await self._queue.put((_STT_END_SENTINEL, frame))
            else:
                logger.warning(
                    "STT(stream): SpeechEndFrame with no active utterance — "
                    "going straight to batch fallback"
                )
                asyncio.create_task(self._batch_fallback._transcribe(frame))
            self._queue = None
            self._worker_task = None

        else:
            await self.push_frame(frame, direction)

    # ── Per-utterance worker: connect, stream chunks, flush, finalize ─────────

    async def _run_utterance(self, queue: asyncio.Queue, connect_t0: float):
        """
        One background task per utterance — owns the entire streaming
        lifecycle so nothing in this class ever blocks process_frame:
          1. Open the WS and start a receiver reading transcript/error events.
          2. Drain queued PCM chunks, sending each to Sarvam as it arrives.
          3. On the end-sentinel (SpeechEndFrame handed off above): send flush,
             wait (bounded) for the post-flush finalized transcript.
          4. On any failure/timeout: fall back to the batch HTTP path using
             the SpeechEndFrame's own complete WAV — same guaranteed-safe path
             SarvamSTTService already uses today.
        """
        params = {
            "language-code": "unknown",
            "model": SARVAM_STT_MODEL,
            "sample_rate": "16000",
            "input_audio_codec": "pcm_s16le",
            # We rely on OUR OWN Silero VAD for utterance boundaries (already
            # tuned — noise floor, barge-in, etc.), not Sarvam's server-side
            # VAD, so vad_signals/high_vad_sensitivity are deliberately left
            # unset. flush_signal=true is what lets our flush force-finalize.
            "flush_signal": "true",
        }
        if SARVAM_STT_MODEL.startswith("saaras"):
            params["mode"] = SARVAM_STT_MODE
        url = f"{self.SARVAM_STT_WS}?{urlencode(params)}"

        ws = None
        stream_failed  = False
        latest_result: dict | None = None
        end_frame: SpeechEndFrame | None = None
        got_result     = asyncio.Event()
        flush_sent_at: float | None = None

        async def receiver():
            """Reads transcript/error events. Only treats a 'data' message as
            FINAL once it arrives after we've sent flush — Sarvam may (rarely)
            emit an earlier data event mid-utterance; we keep the latest one
            but only stop waiting once it's the post-flush result."""
            nonlocal latest_result, stream_failed
            async for raw in ws:
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning(f"STT(stream): non-JSON message ignored: {raw[:200]!r}")
                    continue
                etype = ev.get("type")
                if etype == "data":
                    data = ev.get("data") or {}
                    latest_result = {
                        "transcript":     data.get("transcript", ""),
                        "language_code":  data.get("language_code"),
                    }
                    if flush_sent_at is not None:
                        got_result.set()
                        return
                elif etype == "error":
                    logger.error(f"STT(stream): error event {json.dumps(ev)[:300]}")
                    stream_failed = True
                    got_result.set()
                    return
                # etype == "events" (VAD signals) — not requested (vad_signals
                # unset above), ignored defensively if any arrive anyway.

        try:
            ws = await websockets.connect(
                url, additional_headers={"Api-Subscription-Key": self._api_key}
            )
            logger.info(
                f"LATENCY stt_stream_connect_ms={int((time.monotonic() - connect_t0) * 1000)}"
            )
            receiver_task = asyncio.create_task(receiver())

            while end_frame is None:
                item = await queue.get()
                if isinstance(item, tuple) and item[0] is _STT_END_SENTINEL:
                    end_frame = item[1]
                    flush_sent_at = time.monotonic()
                    await ws.send(json.dumps({"type": "flush"}))
                    break
                await ws.send(json.dumps({
                    "audio": {
                        "data":         base64.b64encode(item).decode("ascii"),
                        "sample_rate":  "16000",
                        "encoding":     "audio/pcm_s16le",
                    }
                }))

            try:
                await asyncio.wait_for(got_result.wait(), timeout=self.FINALIZE_TIMEOUT_SECS)
            except asyncio.TimeoutError:
                logger.warning("STT(stream): finalize timed out — falling back to batch")
                stream_failed = True

            if not receiver_task.done():
                receiver_task.cancel()
            try:
                await receiver_task
            except (asyncio.CancelledError, Exception):
                pass

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Connect failed, or the socket dropped mid-utterance (ws.send
            # raised inside the while-loop above). Either way: drain the
            # queue until we find the end-sentinel so we still have the
            # SpeechEndFrame's WAV bytes to fall back on — the alternative
            # (bailing out early) would silently lose this utterance.
            logger.warning(f"STT(stream): connection failed/dropped (falling back to batch): {exc}")
            stream_failed = True
            while end_frame is None:
                item = await queue.get()
                if isinstance(item, tuple) and item[0] is _STT_END_SENTINEL:
                    end_frame = item[1]
        finally:
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass

        if not stream_failed and latest_result is not None:
            finalize_ms = int((time.monotonic() - flush_sent_at) * 1000)
            logger.info(f"LATENCY stt_stream_finalize_ms={finalize_ms}")
            logger.info("STT: streaming path used")
            transcript = latest_result["transcript"].strip()
            raw_lang   = latest_result.get("language_code") or "en"
            await _finalize_transcript(self.push_frame, transcript, raw_lang, confidence=None)
        else:
            logger.warning("STT: falling back to batch HTTP path for this utterance")
            await self._batch_fallback._transcribe(end_frame)

    async def cleanup(self):
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
        await self._batch_fallback.cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GroqLangGraphProcessor  (LLM + Memory via LangGraph)
# ─────────────────────────────────────────────────────────────────────────────
#
# NEW in this version:
#   Feature 3 (Streaming TTS): uses stream_agent() instead of run_agent().
#     Receives sentences one at a time as the LLM generates them and pushes
#     each as a separate TextFrame to TTS immediately. First audio can start
#     playing before the LLM has finished generating the full response.
#   Feature 6 (Typing Indicator): emits AIThinkingFrame(True) when processing
#     starts and AIThinkingFrame(False) when the first sentence is ready.
#   Feature 10 (Emotion): listens for EmotionHintFrame from STT/VAD and stores
#     the hint. Passes it to stream_agent() where it modifies the system prompt.
#
# Receives:  TranscriptionFrame  (user's transcribed speech)
#            EmotionHintFrame    (from VAD or STT — consumed here)
# Emits:     AIThinkingFrame     (thinking=True/False)
#            TextFrame           (one per LLM sentence → goes to TTS)
#            TranscriptDisplayFrame (full AI reply → browser chat UI)
#

class GroqLangGraphProcessor(FrameProcessor):
    """
    Streaming LLM processor that delegates to LangGraph for memory management.
    Sentences are emitted as they arrive from the LLM stream, enabling
    the TTS pipeline to start synthesizing immediately.
    """

    def __init__(
        self,
        thread_id:   str,
        tts_service  = None,
        session_id:  Optional[str] = None,
        trace_store  = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._thread_id    = thread_id
        # Thread ids superseded by reset_thread() this connection — deleted from
        # the checkpointer on disconnect so their in-RAM state doesn't leak.
        self._retired_thread_ids: list[str] = []
        self._emotion_hint = "neutral"
        self._language     = "en-IN"
        self._tts          = tts_service
        # Phase 3: trace recording
        self._session_id   = session_id or str(uuid.uuid4())
        self._trace_store  = trace_store   # ExecutionTraceStore | None
        self._turn_index   = 0
        # Serialize _generate so the detached opening greeting and a user turn
        # that arrives during it can't run concurrently and corrupt shared
        # per-turn LLM/TTS state (_emotion_hint, _turn_index, TTS turn machinery)
        # (Bug #15).
        self._generate_lock = asyncio.Lock()
        # Resume-on-false-barge-in: the most recently completed turn's
        # sentences, kept around in case its audio gets wiped by a barge-in
        # that turns out to be background noise, not real speech. Set at the
        # end of every completed turn; only ACTED on (replayed or discarded)
        # once mark_turn_cancelled() + resolve_pending_interrupt() run — see
        # those methods for the full flow.
        self._last_turn:     dict | None = None
        self._pending_resume: bool       = False
        logger.info(
            f"GroqLangGraphProcessor: thread_id={thread_id} "
            f"session_id={self._session_id[:8]}… trace={'on' if trace_store else 'off'}"
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # A real, usable transcript arrived — whatever barge-in was
            # pending (if any) is now CONFIRMED real speech, not noise. Drop
            # the cancelled turn we were holding onto for possible resume.
            await self.resolve_pending_interrupt(real_speech=True)
            await self._generate(frame.text)

        elif isinstance(frame, NoSpeechDetectedFrame):
            # STT produced nothing usable. If a turn is awaiting resolution
            # (its audio was just wiped by a barge-in), that barge-in was a
            # false positive — resume it. No-op if nothing is pending (e.g.
            # ordinary background noise between turns, nothing was cancelled).
            await self.resolve_pending_interrupt(real_speech=False)

        elif isinstance(frame, EmotionHintFrame):
            # Feature 10: store emotion hint, use it in next LLM call.
            # Don't push downstream — this frame is consumed here.
            self._emotion_hint = frame.hint
            logger.debug(f"LLM: emotion hint updated → {frame.hint!r}")

        elif isinstance(frame, LanguageDetectedFrame):
            # Capture the detected language so the next LLM call replies in it.
            # Still forward downstream — TTS and browser badge need this frame too.
            self._language = frame.language_code
            logger.debug(f"LLM: language updated → {frame.language_code!r}")
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    def mark_turn_cancelled(self) -> None:
        """
        Called by VoicePipelineManager.interrupt() (barge-in or manual stop)
        right after it wipes the current turn's not-yet-delivered audio. Flags
        the most recently completed turn (_last_turn) as awaiting resolution:
        resolve_pending_interrupt() decides whether to resume it (false
        positive) or discard it (genuine interruption) once STT reports back.
        A no-op if no turn has completed yet (e.g. interrupt during silence).
        """
        if self._last_turn is not None:
            self._pending_resume = True
            logger.debug("LLM: turn cancellation noted — awaiting STT to confirm real speech vs. false positive")

    async def resolve_pending_interrupt(self, real_speech: bool) -> None:
        """
        Resolve a turn cancellation once we know whether the interruption was
        real. real_speech=True (a usable transcript arrived) → discard the
        cancelled turn, the new input takes over. real_speech=False (STT
        found nothing usable) → the "interruption" was background noise, not
        the user talking: resume the cancelled turn by re-emitting its
        sentences as fresh TextFrames, exactly as if it had never been cut off.
        No-op if no turn is currently awaiting resolution.
        """
        if not self._pending_resume or self._last_turn is None:
            return
        self._pending_resume = False
        turn, self._last_turn = self._last_turn, None

        if real_speech:
            logger.info("LLM: barge-in confirmed as real user speech — discarding the cancelled turn")
            return

        sentences = turn["sentences"]
        logger.info(
            f"LLM: barge-in was a false positive (no real speech followed) — "
            f"resuming {len(sentences)} cancelled sentence(s)"
        )
        # The frontend closes its audio-playback gate on barge_in/interrupted
        # and only reopens it on a fresh {"type":"thinking","active":true}
        # message (see useWebSocket.ts's acceptAudioRef) — otherwise it
        # silently drops audio from a "cancelled" turn so a stale reply can't
        # resume by mistake. Without re-sending that signal here, THIS
        # legitimate resume would be dropped client-side even though the
        # server delivered it correctly.
        await self.push_frame(AIThinkingFrame(thinking=True))
        await self.push_frame(AIThinkingFrame(thinking=False))
        for sentence in sentences:
            await self.push_frame(TextFrame(text=sentence))
        await self.push_frame(
            TranscriptDisplayFrame(text=" ".join(sentences), speaker="ai")
        )
        await self.push_frame(LLMTurnDoneFrame())
        if turn.get("end_call"):
            await self.push_frame(EndCallFrame())

    async def _generate(self, user_text: str):
        """
        Serialized entry point for a turn. The lock ensures the detached opening
        greeting and any user turn that arrives during it run one-at-a-time, so
        they can't interleave TextFrames into the same TTS turn or race the
        shared _emotion_hint/_turn_index state (Bug #15).
        """
        async with self._generate_lock:
            await self._generate_impl(user_text)

    async def _generate_impl(self, user_text: str):
        """
        Stream user text through LangGraph + Groq, emitting one TextFrame
        per sentence for immediate TTS synthesis.

        Phase 3: wraps the call with wall-clock timing and records a TurnTrace
        into Qdrant after every turn via ExecutionTraceStore.

        Flow:
          1. Emit AIThinkingFrame(True)
          2. Start timer
          3. Call stream_agent() — RAG context injected inside stream_agent (Phase 2)
          4. On first sentence: emit AIThinkingFrame(False)
          5. Push each sentence as TextFrame → TTS
          6. In finally: emit TranscriptDisplayFrame + record TurnTrace
          7. Reset emotion_hint to neutral
          8. Push LLMTurnDoneFrame sentinel
        """
        from app.langgraph_flow import stream_agent
        from app.tracing.trace_store import TurnTrace

        logger.info(f"LangGraph streaming: thread={self._thread_id[:8]}… input={user_text!r}")

        await self.push_frame(AIThinkingFrame(thinking=True))

        full_text      = ""
        sentences: list[str] = []    # mirrors full_text, for resume-on-false-barge-in
        first_sentence = True
        meta_out: dict = {}          # Phase 2+3: populated by stream_agent
        start_time     = time.monotonic()
        llm_first_ms: int | None = None   # LATENCY: turn start → first sentence
        # Stamp turn start on the TTS service so it can measure the time to the
        # FIRST audio byte (tts_first_audio_ms) — the number that governs how
        # responsive the agent feels. Turns are serialized by _generate_lock and
        # TTS processes one turn at a time, so this single field is race-free.
        if self._tts is not None:
            self._tts._turn_t0             = start_time
            self._tts._first_audio_recorded = False

        try:
            async for sentence in stream_agent(
                user_text,
                self._thread_id,
                self._emotion_hint,
                self._language,
                _meta_out=meta_out,
            ):
                if not sentence.strip():
                    continue

                if first_sentence:
                    llm_first_ms = int((time.monotonic() - start_time) * 1000)
                    logger.info(f"LATENCY llm_first_sentence_ms={llm_first_ms}")
                    await self.push_frame(AIThinkingFrame(thinking=False))
                    first_sentence = False

                full_text += sentence.strip() + " "
                sentences.append(sentence.strip())
                logger.info(f"LLM→TTS: pushing TextFrame → {sentence.strip()!r}")
                await self.push_frame(TextFrame(text=sentence.strip()))

        except Exception as e:
            logger.error(f"LangGraph streaming error: {e}")

        finally:
            latency_ms = int((time.monotonic() - start_time) * 1000)

            if first_sentence:
                await self.push_frame(AIThinkingFrame(thinking=False))

            if full_text.strip():
                await self.push_frame(TranscriptDisplayFrame(text=full_text.strip(), speaker="ai"))

            # Phase 3: record turn trace
            if self._trace_store is not None:
                try:
                    trace = TurnTrace(
                        session_id        = self._session_id,
                        turn_index        = self._turn_index,
                        user_input        = user_text,
                        detected_language = self._language,
                        retrieved_docs    = meta_out.get("retrieved_docs", []),
                        tool_calls        = meta_out.get("tool_calls",     []),
                        ai_response       = full_text.strip(),
                        latency_ms        = latency_ms,
                        llm_first_ms      = llm_first_ms,
                        emotion_hint      = self._emotion_hint,
                        created_at        = datetime.now(timezone.utc).isoformat(),
                    )
                    # Bound the Qdrant upsert: it sits on the turn-completion
                    # path just before LLMTurnDoneFrame/EndCallFrame, so a slow
                    # Qdrant would inject dead time between reply-end and
                    # turn/hangup completion (Bug #14). Cap it at 1s; on timeout
                    # we skip persistence for this turn rather than stall the call.
                    self._turn_index += 1
                    await asyncio.wait_for(self._trace_store.record_turn(trace), timeout=1.0)
                except asyncio.TimeoutError:
                    logger.warning("TraceStore.record_turn timed out (>1s) — skipping trace for this turn")
                except Exception as trace_exc:
                    logger.error(f"TraceStore.record_turn failed (non-fatal): {trace_exc}")

        self._emotion_hint = "neutral"

        # Resume-on-false-barge-in: remember this turn so mark_turn_cancelled()
        # + resolve_pending_interrupt() can replay it if a barge-in wipes its
        # audio and then turns out to have been background noise, not real
        # speech (see those methods above for the full flow).
        if full_text.strip():
            self._last_turn = {"sentences": sentences, "end_call": bool(meta_out.get("end_call"))}

        logger.info("LLM: pushing LLMTurnDoneFrame sentinel downstream")
        await self.push_frame(LLMTurnDoneFrame())

        # Agent-initiated hangup: if the LLM signalled [END_CALL] this turn,
        # push EndCallFrame AFTER LLMTurnDoneFrame so it arrives in-order behind
        # the goodbye audio. The TTS service forwards it to OutputSink only once
        # this turn's audio has been delivered, so the farewell is always heard.
        if meta_out.get("end_call"):
            logger.info("LLM: agent requested end-of-call — pushing EndCallFrame")
            await self.push_frame(EndCallFrame())

    def reset_thread(self):
        """Start a new conversation by assigning a new thread_id."""
        old = self._thread_id
        # Remember the retired thread so its checkpointer state is deleted on
        # disconnect — otherwise reset() leaks the old conversation in RAM.
        self._retired_thread_ids.append(old)
        self._thread_id = str(uuid.uuid4())
        logger.info(f"LangGraph: thread reset {old[:8]}→{self._thread_id[:8]}")

    async def cleanup_threads(self):
        """
        Delete this connection's checkpointer state (current + any retired
        thread ids) on disconnect so the global in-RAM MemorySaver doesn't grow
        without bound. Non-fatal: a failed delete is logged, not raised.
        """
        from app.memory import checkpointer
        for tid in [*self._retired_thread_ids, self._thread_id]:
            try:
                await checkpointer.adelete_thread(tid)
            except Exception as exc:
                logger.warning(f"cleanup_threads: delete {tid[:8]}… failed (non-fatal): {exc}")
        self._retired_thread_ids.clear()


# ─────────────────────────────────────────────────────────────────────────────
# TTS pronunciation normalisation
# ─────────────────────────────────────────────────────────────────────────────
# (regex, replacement) pairs applied to text JUST before it goes to Sarvam TTS.
# Fixes brand/product names the TTS mispronounces. Spoken-only — transcript and
# logs keep the original spelling. \b = whole-word, re.IGNORECASE = any casing.
#
# To add a term: append (re.compile(r"\bWORD\b", re.IGNORECASE), "phonetic").
#
# WHY DEVANAGARI: Sarvam bulbul:v2 with target_language_code="en-IN" applies
# English grapheme-to-phoneme rules to Roman text, so "Bharat"/"Suhas" get
# anglicised (wrong aspiration + vowels). Feeding the word in Devanagari routes
# it through the Indic phoneme set and it reads natively — even inside an
# otherwise-English sentence (mixed-script input is supported; verified by ear).
_TTS_PRONUNCIATION_MAP = [
    # Brand: "BharatConnect"/"Bharat Connect"/"BharatConnect's" → full Devanagari
    # "भारत कनेक्ट". Keeping "Connect" in Roman mid-phrase made the TTS switch out
    # of Indic mode and garble it; writing the WHOLE brand in one script fixes the
    # pronunciation (verified by ear).
    (re.compile(r"\bBharat[\s-]?Connect\b", re.IGNORECASE), "भारत कनेक्ट"),
    # Agent name — Devanagari gives the correct "su-HAAS", not English "soo-HASS".
    (re.compile(r"\bSuha+s\b", re.IGNORECASE), "सुहास"),
    # Competitors — Devanagari for native pronunciation.
    (re.compile(r"\bJio\b", re.IGNORECASE), "जियो"),
    (re.compile(r"\bAirtel\b", re.IGNORECASE), "एयरटेल"),
    (re.compile(r"\bVodafone\b", re.IGNORECASE), "वोडाफ़ोन"),
    (re.compile(r"\bTeleNova\b", re.IGNORECASE), "टेलीनोवा"),
    (re.compile(r"\bBSNL\b"), "B S N L"),                    # spell the acronym
    (re.compile(r"\bVi\b"), "V I"),                          # Vi (Vodafone Idea) → "V I"
    (re.compile(r"\bVoLTE\b", re.IGNORECASE), "V O L T E"),  # acronym — spell it, don't say "volte"
    # Cities — Devanagari for correct Indic pronunciation.
    (re.compile(r"\bBengaluru\b", re.IGNORECASE), "बेंगलुरु"),
    (re.compile(r"\bChennai\b", re.IGNORECASE), "चेन्नई"),
    (re.compile(r"\bBihar\b", re.IGNORECASE), "बिहार"),
    (re.compile(r"\bDelhi\b", re.IGNORECASE), "दिल्ली"),     # "Delhi NCR" → "दिल्ली NCR" (NCR stays Roman)
    # CEO name.
    (re.compile(r"\bAnanya\s+Deshpande\b", re.IGNORECASE), "अनन्या देशपांडे"),
    # Terms.
    (re.compile(r"\bAadhaar\b", re.IGNORECASE), "आधार"),
    # Product term: British "fibre" sometimes reads oddly; normalise to "fiber".
    # Also the standalone product word "Fiber" reads better in Devanagari.
    (re.compile(r"\bfib(?:re|er)\b", re.IGNORECASE), "फ़ाइबर"),
]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SarvamTTSService  (Text-to-Speech)
# ─────────────────────────────────────────────────────────────────────────────
#
# NEW in this version:
#   Feature 3 (Streaming TTS): now receives one TextFrame per sentence
#     (instead of one big TextFrame) and synthesizes each immediately.
#     Since the TTS API handles short sentences faster, the first audio
#     chunk reaches the browser much sooner.
#   Feature 7 (Language Auto-Switch): listens for LanguageDetectedFrame
#     and updates the target_language_code used in all subsequent API calls.
#
# Receives:  TextFrame              (one sentence of AI reply)
#            LanguageDetectedFrame  (updates TTS language)
# Emits:     AIStatusFrame(True)   (before first sentence audio)
#            AIAudioFrame           (WAV bytes for one sentence)
#            AIStatusFrame(False)   (after last sentence audio)
#

def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV header.

    The Sarvam TTS WebSocket returns raw PCM (content_type audio/pcm); the
    browser client and the batch path both expect self-contained WAV blobs.
    Wrapping each streamed chunk here keeps the downstream AIAudioFrame contract
    (and main.py's SARVAM_AUDIO_BYTES_PER_SEC math) byte-identical to today, so
    nothing downstream — server or browser — needs to change.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _concat_wavs(wav_list: list[bytes]) -> bytes:
    """
    Concatenate multiple WAV byte blobs into a single WAV.
    All clips must share the same sample rate, channels, and sample width
    (Sarvam always returns mono 22050 Hz 16-bit, so this is safe).
    Returns a single well-formed WAV byte string.
    """
    if not wav_list:
        return b""
    if len(wav_list) == 1:
        return wav_list[0]

    frames_list = []
    params = None
    for wav_bytes in wav_list:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            if params is None:
                params = wf.getparams()
            frames_list.append(wf.readframes(wf.getnframes()))

    out_buf = io.BytesIO()
    with wave.open(out_buf, "wb") as wf_out:
        wf_out.setparams(params)
        for frames in frames_list:
            wf_out.writeframes(frames)
    return out_buf.getvalue()


class SarvamTTSService(FrameProcessor):
    """
    Synthesizes speech chunk by chunk with CONCURRENT API calls + in-order delivery.

    Problem with the old design:
      process_frame() awaited _synthesize() before returning, so the pipeline
      stalled waiting for the TTS HTTP round-trip (~300-500 ms) before the next
      TextFrame could even be pushed. Groq streaming was effectively serialised
      through the TTS bottleneck — the user heard nothing until the last chunk
      was synthesised.

    New design — pipeline-parallel TTS:
      1. Each TextFrame immediately fires a background asyncio.Task for the API
         call. process_frame() returns right away so Groq can keep streaming.
      2. A monotonic sequence counter (_seq) stamps each chunk in arrival order.
      3. A delivery loop (_delivery_task) waits for futures in order and pushes
         AIAudioFrame to OutputSink the moment each one resolves, preserving
         playback order even if a later chunk's API call finishes first.
      4. When the LLM turn ends, GroqLangGraphProcessor pushes a TextFrame with
         text=None as a sentinel — the delivery loop drains remaining futures,
         emits AIStatusFrame(False), and resets for the next turn.
    """

    SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
    TTS_CHAR_LIMIT  = 450

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self._api_key          = api_key
        self._http             = httpx.AsyncClient(timeout=30.0)
        self._language         = "en-IN"   # Feature 7: updated by LanguageDetectedFrame
        self._tts_active       = False
        # Agent-initiated hangup: set True when an EndCallFrame arrives for the
        # current turn. The delivery loop emits EndCallFrame downstream only
        # AFTER the last audio chunk, so the goodbye is heard before the hangup.
        self._end_call_pending = False
        # In-order delivery: list of futures in arrival order
        self._pending: list    = []
        self._delivery_task    = None
        # Lazily created in process_frame (needs a running loop)
        self._llm_done: asyncio.Event | None = None
        # Track background tasks so interrupt() can cancel them
        self._inflight_tasks: list = []
        # LATENCY: turn start (monotonic secs) stamped by the LLM processor at
        # turn start; the first-audio delta is recorded once per turn.
        self._turn_t0: float | None = None
        self._first_audio_recorded  = False

    # ── Pipecat frame handler ─────────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            loop = asyncio.get_running_loop()
            fut  = loop.create_future()
            # Capture language NOW — before any future LanguageDetectedFrame
            # could arrive mid-turn and change self._language underneath us.
            lang_at_dispatch = self._language

            # Detect a NEW turn by checking whether the previous turn's
            # llm_done event is already set (or no event exists yet).
            # Using _delivery_task.done() is NOT reliable because the task
            # stays alive through its finally-block (which awaits push_frame),
            # and TextFrames from the new turn can arrive during that await —
            # they would be appended to the old _pending list and then wiped
            # by my_pending.clear() in the finally block.
            # Checking llm_done.is_set() is the correct signal: it means the
            # previous turn has fully committed all its sentences, so any new
            # TextFrame belongs to a fresh turn that needs its own list+event.
            new_turn = (self._llm_done is None or self._llm_done.is_set())
            if new_turn:
                self._llm_done      = asyncio.Event()   # fresh, unset
                self._pending       = []                # fresh list for this turn
                self._delivery_task = asyncio.create_task(self._deliver_in_order())

            self._pending.append(fut)
            task = asyncio.create_task(
                self._call_tts_api(frame.text, fut, lang_at_dispatch)
            )
            self._inflight_tasks.append(task)
            task.add_done_callback(lambda t: self._inflight_tasks.remove(t)
                                   if t in self._inflight_tasks else None)

        elif isinstance(frame, LLMTurnDoneFrame):
            # All TextFrames for this turn have now been processed in order.
            # Just set the event — do NOT await flush() here because that would
            # block process_frame while the delivery loop tries to push frames
            # back through the same pipeline, causing a deadlock.
            # The delivery loop drains itself once _llm_done is set.
            logger.info("TTS: LLMTurnDoneFrame received — setting _llm_done")
            if self._llm_done is not None:
                self._llm_done.set()

        elif isinstance(frame, EndCallFrame):
            # Do NOT forward yet — the goodbye audio is still being delivered
            # asynchronously by _deliver_in_order(). Mark it pending; the
            # delivery loop emits EndCallFrame after the final audio chunk.
            logger.info("TTS: EndCallFrame received — will hang up after goodbye audio")
            self._end_call_pending = True

        elif isinstance(frame, LanguageDetectedFrame):
            self._language = frame.language_code
            logger.info(f"TTS: language switched to {frame.language_code!r}")
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    # ── Background: HTTP call, resolves a future with wav bytes ──────────────

    async def _call_tts_api(self, text: str, fut: asyncio.Future, language: str):
        """Call Sarvam TTS and resolve *fut* with WAV bytes (or None on error)."""
        chunk_id = id(fut)
        try:
            # Spell out digits BEFORE pronunciation/truncation: Sarvam drops or
            # mis-speaks bare numerals (esp. Devanagari "६५"), so convert them to
            # words in this turn's language so prices are always heard.
            spoken = _spell_digits(text, language)
            tts_text = self._truncate(self._normalize_pronunciation(spoken))
            logger.info(f"TTS[{chunk_id}]: START request text={tts_text!r} lang={language!r}")

            resp = await self._http.post(
                self.SARVAM_TTS_URL,
                headers={
                    "api-subscription-key": self._api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "inputs":               [tts_text],
                    "target_language_code": language,
                    "speaker":              SARVAM_TTS_SPEAKER,
                    "model":                SARVAM_TTS_MODEL,
                },
            )

            if resp.status_code != 200:
                logger.error(f"TTS[{chunk_id}]: HTTP {resp.status_code}: {resp.text[:200]}")
                if not fut.done():
                    fut.set_result(None)
                return

            audios_b64 = resp.json().get("audios", [])
            audios_b64 = [a for a in audios_b64 if a]  # drop empty entries
            if not audios_b64:
                logger.error(f"TTS[{chunk_id}]: empty audio in response")
                if not fut.done():
                    fut.set_result(None)
                return

            if len(audios_b64) == 1:
                wav_bytes = base64.b64decode(audios_b64[0])
            else:
                # Sarvam may return multiple audio clips (e.g. when the input
                # contains a comma it sometimes splits internally).
                # Concatenate all clips into a single WAV so nothing is lost.
                wav_bytes = _concat_wavs([base64.b64decode(a) for a in audios_b64])
                logger.info(f"TTS[{chunk_id}]: concatenated {len(audios_b64)} audio clips")

            logger.info(f"TTS[{chunk_id}]: DONE resolved {len(wav_bytes)} bytes for {tts_text!r}")
            if not fut.done():
                fut.set_result(wav_bytes)

        except asyncio.CancelledError:
            logger.warning(f"TTS[{chunk_id}]: CANCELLED for text={text!r}")
            if not fut.done():
                fut.set_result(None)
        except Exception as e:
            logger.error(f"TTS[{chunk_id}]: ERROR {e} for text={text!r}")
            if not fut.done():
                fut.set_result(None)

    # ── Delivery loop: push audio frames in original arrival order ────────────

    async def _deliver_in_order(self):
        """
        Drain _pending futures in order, pushing audio downstream as each resolves.

        The loop keeps running until BOTH conditions are true:
          1. _llm_done is set  (flush() called after LLM finishes streaming)
          2. _pending is empty (all in-flight TTS requests have resolved)

        This prevents the race where the loop exits after draining an early batch
        while the LLM is still yielding more chunks.
        """
        chunk_n = 0
        first   = True
        # Take a snapshot of BOTH the event AND the pending list for THIS turn.
        # If a new turn starts while we're in finally-block cleanup,
        # process_frame will have already replaced self._llm_done and
        # self._pending — we must not touch those new-turn objects here.
        llm_done     = self._llm_done
        my_pending   = self._pending   # same list object for this turn
        try:
            while True:
                if not my_pending:
                    # Nothing left to deliver. Exit only if the LLM has
                    # confirmed it's done sending text for THIS turn.
                    if llm_done.is_set():
                        break
                    # LLM still streaming — wait for it to signal done,
                    # then loop back to check my_pending again (more chunks
                    # may have arrived while we were waiting).
                    try:
                        await asyncio.wait_for(llm_done.wait(), timeout=30.0)
                    except asyncio.TimeoutError:
                        logger.error("TTS delivery: timed out waiting for LLM done signal")
                        break
                    continue

                fut = my_pending.pop(0)
                chunk_n += 1
                fut_id = id(fut)
                logger.info(f"TTS delivery: waiting for chunk #{chunk_n} fut={fut_id}")
                try:
                    wav_bytes = await asyncio.wait_for(
                        asyncio.shield(fut), timeout=15.0
                    )
                except asyncio.TimeoutError:
                    logger.error(f"TTS delivery: chunk #{chunk_n} timed out, skipping")
                    continue

                if wav_bytes is None:
                    logger.error(f"TTS delivery: chunk #{chunk_n} fut={fut_id} resolved None — TTS API failed, skipping")
                    continue

                logger.info(f"TTS delivery: pushing chunk #{chunk_n} ({len(wav_bytes)} bytes) to browser")
                if first:
                    first = False
                    self._tts_active = True
                    # LATENCY: time from LLM turn start to the FIRST audio byte
                    # leaving for the browser — the single number that governs how
                    # responsive the agent feels. Measured HERE because, with the
                    # pipeline-parallel TTS design, first-audio happens
                    # asynchronously relative to the LLM turn's own bookkeeping,
                    # so it cannot be captured in the LLM processor's TurnTrace.
                    if self._turn_t0 is not None and not self._first_audio_recorded:
                        self._first_audio_recorded = True
                        tts_first_audio_ms = int((time.monotonic() - self._turn_t0) * 1000)
                        logger.info(f"LATENCY tts_first_audio_ms={tts_first_audio_ms}")
                    await self.push_frame(AIStatusFrame(ai_speaking=True))

                await self.push_frame(AIAudioFrame(audio_bytes=wav_bytes))

        except asyncio.CancelledError:
            logger.warning("TTS delivery: cancelled mid-delivery")
            pass  # interrupted — finally block handles cleanup
        except Exception as e:
            logger.error(f"TTS delivery error: {e}")
        finally:
            logger.info(f"TTS delivery: loop exiting — delivered {chunk_n} chunks, tts_active={self._tts_active}")
            # Emit AIStatusFrame(False) if we EVER signalled speaking (normal case)
            # OR if this turn processed chunks but delivered no audio at all
            # (total TTS failure — every chunk resolved None). Without the latter,
            # a fully-failed turn emitted no status frame and the client was left
            # stuck showing "thinking" with no degradation cue (Bug #16). `first`
            # is still True iff nothing was ever delivered.
            total_failure = first and chunk_n > 0
            if self._tts_active or total_failure:
                if total_failure:
                    logger.warning("TTS delivery: all chunks failed — emitting AIStatusFrame(False) so client recovers")
                await self.push_frame(AIStatusFrame(ai_speaking=False))
            self._tts_active = False
            # Clear only this turn's pending list — NOT self._pending, which
            # may have already been replaced by a new turn's list object.
            my_pending.clear()

            # Agent-initiated hangup: now that the goodbye audio has been fully
            # delivered, forward EndCallFrame so main.py closes the WebSocket.
            if self._end_call_pending:
                self._end_call_pending = False
                logger.info("TTS delivery: goodbye delivered — forwarding EndCallFrame downstream")
                await self.push_frame(EndCallFrame())

    # ── Called by GroqLangGraphProcessor when the LLM turn is complete ────────

    async def flush(self):
        """
        Signal that the LLM has finished streaming, then wait for the delivery
        loop to drain all remaining in-flight TTS requests and push their audio.
        """
        if self._llm_done is not None:
            self._llm_done.set()
        if self._delivery_task and not self._delivery_task.done():
            try:
                await asyncio.wait_for(self._delivery_task, timeout=30.0)
            except asyncio.TimeoutError:
                logger.error("TTS flush: timed out")
                self._delivery_task.cancel()
            except Exception as e:
                logger.error(f"TTS flush error: {e}")

    async def cancel_turn(self):
        """
        Called on barge-in / interrupt: cancel all in-flight TTS API tasks
        and stop the delivery loop immediately so no stale audio is pushed.
        """
        # Cancel in-flight HTTP tasks
        for task in list(self._inflight_tasks):
            task.cancel()
        self._inflight_tasks.clear()

        # Resolve any pending futures with None so the delivery loop unblocks
        for fut in self._pending:
            if not fut.done():
                fut.set_result(None)
        self._pending.clear()

        # Cancel the delivery loop itself
        if self._delivery_task and not self._delivery_task.done():
            self._delivery_task.cancel()
            try:
                await self._delivery_task
            except (asyncio.CancelledError, Exception):
                pass

        self._tts_active = False
        # Reset to the pristine "no turn in progress" state so the NEXT TextFrame
        # is detected as a fresh turn (new_turn keys off _llm_done is None or
        # is_set() in process_frame). Clearing it instead would leave it unset-but-
        # not-None, so the next turn would be mistaken for a continuation of this
        # cancelled one and its audio would never be delivered — the bug that made
        # the agent go silent after a barge-in.
        self._llm_done      = None
        self._pending       = []
        self._delivery_task = None
        # LATENCY: drop this turn's timing so a barged-in turn never leaks its
        # start into the next turn's tts_first_audio_ms.
        self._turn_t0              = None
        self._first_audio_recorded = False

    def _normalize_pronunciation(self, text: str) -> str:
        """
        Rewrite hard-to-pronounce brand/product terms into phonetic spellings
        Sarvam TTS handles cleanly. This affects ONLY the spoken audio — the
        displayed transcript and logs keep the original spelling.

        The model often writes the brand as "BharatConnect" (one camelCase word)
        which the TTS mangles; "Bharat Connect" (two words) reads correctly.
        Competitor names and "fibre/fiber" are normalised the same way.
        Case-insensitive, whole-word matches only.
        """
        for pattern, replacement in _TTS_PRONUNCIATION_MAP:
            text = pattern.sub(replacement, text)
        return text

    def _truncate(self, text: str) -> str:
        """Keep text within Sarvam's per-item character limit."""
        if len(text) <= self.TTS_CHAR_LIMIT:
            return text
        truncated = text[:self.TTS_CHAR_LIMIT]
        for punct in (".", "?", "!"):
            last = truncated.rfind(punct)
            if last > self.TTS_CHAR_LIMIT // 2:
                return truncated[:last + 1]
        return truncated

    async def cleanup(self):
        await self._http.aclose()


# ─────────────────────────────────────────────────────────────────────────────
# 4b. SarvamTTSStreamingService  (Text-to-Speech over the streaming WebSocket)
# ─────────────────────────────────────────────────────────────────────────────
#
# Flag-gated alternative to SarvamTTSService (batch HTTP), enabled by
# TTS_STREAMING=true. Chosen in VoicePipelineManager; when the flag is off this
# class is never instantiated and behavior is byte-identical to before.
#
# Why it can be a drop-in: the Sarvam TTS WebSocket returns raw 22050 Hz mono
# PCM. We wrap each streamed chunk in a WAV header (_pcm_to_wav) and emit the
# SAME AIAudioFrame(WAV) the batch path emits, so OutputSink, main.py's
# send_loop, and the browser client are all unchanged.
#
# Lifecycle is PER-TURN (like the harness): a worker task opens the WS on the
# first TextFrame, a sender streams sentence text in as it arrives, and a reader
# loop wraps audio chunks and pushes them downstream until Sarvam's end signal
# ({"type":"event","data":{"event_type":"final"}}), then closes the socket —
# which is what avoids the idle "408 left open too long" the spike surfaced.
#
# Receives:  TextFrame, LLMTurnDoneFrame, EndCallFrame, LanguageDetectedFrame
# Emits:     AIStatusFrame(True/False), AIAudioFrame(WAV per chunk), EndCallFrame
#

class SarvamTTSStreamingService(FrameProcessor):
    """Streaming Sarvam TTS over a per-turn WebSocket. See section header."""

    SARVAM_TTS_WS   = "wss://api.sarvam.ai/text-to-speech/ws"
    TTS_CHAR_LIMIT  = 450
    MODEL           = SARVAM_TTS_MODEL
    SPEAKER         = SARVAM_TTS_SPEAKER
    SAMPLE_RATE     = 22050   # matches main.py SARVAM_AUDIO_BYTES_PER_SEC

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self._api_key      = api_key
        self._language     = "en-IN"   # Feature 7: updated by LanguageDetectedFrame
        self._tts_active   = False
        self._end_call_pending = False
        # Per-turn worker + its text queue. A turn "accepts text" from its first
        # TextFrame until LLMTurnDoneFrame flips _turn_flushed; the next TextFrame
        # then starts a fresh turn (its own queue, worker, and WebSocket).
        self._text_queue: asyncio.Queue | None = None
        self._worker_task: asyncio.Task | None = None
        self._turn_flushed = False
        # LATENCY: turn start (monotonic secs) stamped by the LLM processor; the
        # first-audio delta is logged once per turn. Kept for parity with the
        # batch service so GroqLangGraphProcessor can stamp either impl.
        self._turn_t0: float | None = None
        self._first_audio_recorded  = False

    # ── Pipecat frame handler ─────────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            new_turn = self._text_queue is None or self._turn_flushed
            if new_turn:
                self._turn_flushed = False
                self._text_queue   = asyncio.Queue()
                # Capture per-turn context now so a later LanguageDetectedFrame
                # or _turn_t0 change can't alter this turn's worker underneath us.
                self._worker_task = asyncio.create_task(
                    self._run_turn(self._text_queue, self._language, self._turn_t0)
                )
            await self._text_queue.put(frame.text)

        elif isinstance(frame, LLMTurnDoneFrame):
            # End of this turn's text. Signal the sender to flush; the reader
            # drains remaining audio and self-terminates on the 'final' event.
            self._turn_flushed = True
            if self._text_queue is not None:
                await self._text_queue.put(None)

        elif isinstance(frame, EndCallFrame):
            # Forwarded by the worker only AFTER the goodbye audio is delivered,
            # so the farewell is heard before main.py closes the socket.
            logger.info("TTS(stream): EndCallFrame received — hang up after goodbye")
            self._end_call_pending = True

        elif isinstance(frame, LanguageDetectedFrame):
            self._language = frame.language_code
            logger.info(f"TTS(stream): language switched to {frame.language_code!r}")
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    # ── Per-turn worker: open WS, stream text in, push audio out ──────────────

    async def _run_turn(self, text_queue: asyncio.Queue, language: str, turn_t0):
        url = f"{self.SARVAM_TTS_WS}?{urlencode({'model': self.MODEL, 'send_completion_event': 'true'})}"
        ws = None
        sender = None
        first_audio  = True
        emitted_true = False   # did we push AIStatusFrame(True) this turn?
        try:
            ws = await websockets.connect(
                url, additional_headers={"Api-Subscription-Key": self._api_key}
            )
            await ws.send(json.dumps({
                "type": "config",
                "data": {
                    "model":                self.MODEL,
                    "target_language_code": language,
                    "speaker":              self.SPEAKER,
                    "output_audio_codec":   "linear16",
                    "speech_sample_rate":   self.SAMPLE_RATE,
                },
            }))
            sender = asyncio.create_task(self._sender(ws, text_queue, language))

            async for raw in ws:
                ev    = json.loads(raw)
                etype = ev.get("type")
                if etype == "audio":
                    b64 = (ev.get("data") or {}).get("audio", "")
                    if not b64:
                        continue
                    wav = _pcm_to_wav(base64.b64decode(b64), self.SAMPLE_RATE)
                    if first_audio:
                        first_audio  = False
                        emitted_true = True
                        self._tts_active = True
                        if turn_t0 is not None:
                            ms = int((time.monotonic() - turn_t0) * 1000)
                            logger.info(f"LATENCY tts_first_audio_ms={ms}")
                        await self.push_frame(AIStatusFrame(ai_speaking=True))
                    await self.push_frame(AIAudioFrame(audio_bytes=wav))
                elif etype == "event" and (ev.get("data") or {}).get("event_type") == "final":
                    break
                elif etype in {"complete", "completed"}:
                    break
                elif etype == "error":
                    logger.error(f"TTS(stream): error event {json.dumps(ev)[:300]}")
                    break

            if sender is not None:
                sender.cancel()

        except asyncio.CancelledError:
            # Barge-in / interrupt. Re-raise after finally cleans up the socket.
            logger.warning("TTS(stream): turn cancelled")
            raise
        except Exception as exc:
            logger.error(f"TTS(stream): turn error: {exc}")
        finally:
            if sender is not None and not sender.done():
                sender.cancel()
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass
            # Mirror the batch path: emit AIStatusFrame(False) if we ever signalled
            # speaking, so the client always leaves the "speaking" state. Guarded
            # because on cancel the push may itself be interrupted.
            if emitted_true:
                try:
                    await self.push_frame(AIStatusFrame(ai_speaking=False))
                except Exception:
                    pass
            self._tts_active = False
            if self._end_call_pending:
                self._end_call_pending = False
                logger.info("TTS(stream): goodbye delivered — forwarding EndCallFrame")
                try:
                    await self.push_frame(EndCallFrame())
                except Exception:
                    pass

    async def _sender(self, ws, text_queue: asyncio.Queue, language: str):
        """Drain the text queue into the WS; a None item means flush + stop."""
        try:
            while True:
                text = await text_queue.get()
                if text is None:
                    await ws.send(json.dumps({"type": "flush"}))
                    return
                # Same pre-TTS processing as the batch path: spell digits in the
                # reply language, apply the pronunciation map, then truncate.
                spoken   = _spell_digits(text, language)
                tts_text = self._truncate(self._normalize_pronunciation(spoken))
                if tts_text.strip():
                    await ws.send(json.dumps({"type": "text", "data": {"text": tts_text}}))
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error(f"TTS(stream): sender error: {exc}")

    # ── Turn control (parity with SarvamTTSService) ───────────────────────────

    async def flush(self):
        """Wait for the current turn's worker to finish draining audio."""
        if self._worker_task and not self._worker_task.done():
            try:
                await asyncio.wait_for(self._worker_task, timeout=30.0)
            except asyncio.TimeoutError:
                logger.error("TTS(stream): flush timed out")
                self._worker_task.cancel()
            except Exception as exc:
                logger.error(f"TTS(stream): flush error: {exc}")

    async def cancel_turn(self):
        """Barge-in / interrupt: kill the worker and its WebSocket immediately."""
        task = self._worker_task
        # Reset to a pristine 'no turn in progress' state so the NEXT TextFrame
        # is treated as a fresh turn.
        self._worker_task = None
        self._text_queue  = None
        self._turn_flushed = False
        self._turn_t0     = None
        self._first_audio_recorded = False
        self._tts_active  = False
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    def _normalize_pronunciation(self, text: str) -> str:
        for pattern, replacement in _TTS_PRONUNCIATION_MAP:
            text = pattern.sub(replacement, text)
        return text

    def _truncate(self, text: str) -> str:
        if len(text) <= self.TTS_CHAR_LIMIT:
            return text
        truncated = text[:self.TTS_CHAR_LIMIT]
        for punct in (".", "?", "!"):
            last = truncated.rfind(punct)
            if last > self.TTS_CHAR_LIMIT // 2:
                return truncated[:last + 1]
        return truncated

    async def cleanup(self):
        await self.cancel_turn()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  OutputSink
# ─────────────────────────────────────────────────────────────────────────────
#
# NEW in this version: forwards the new frame types (AIThinkingFrame,
# LanguageDetectedFrame, BargeInDetectedFrame) to the output queue so that
# main.py can handle them.
#

class OutputSink(FrameProcessor):
    """
    Last processor in the pipeline. Puts output frames onto an asyncio.Queue
    that main.py reads from to send data back to the browser.

    Handles:
      TranscriptDisplayFrame  → browser chat display
      AIAudioFrame            → browser audio playback
      AIStatusFrame           → speaking indicator
      AIThinkingFrame         → Feature 6: thinking indicator
      LanguageDetectedFrame   → Feature 7: language badge
      BargeInDetectedFrame    → Feature 4: auto-interrupt signal
      EndFrame                → signals pipeline shutdown
    """

    def __init__(self, output_queue: asyncio.Queue, **kwargs):
        super().__init__(**kwargs)
        self._q = output_queue

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Put all output-relevant frames onto the queue for main.py
        if isinstance(frame, (
            TranscriptDisplayFrame,
            AIAudioFrame,
            AIStatusFrame,
            AIThinkingFrame,        # Feature 6
            LanguageDetectedFrame,  # Feature 7
            BargeInDetectedFrame,   # Feature 4
            EndCallFrame,           # Agent-initiated hangup
            EndFrame,
        )):
            await self._q.put(frame)

        # CRITICAL: always push ALL frames downstream, including system frames
        # (StartFrame, StopFrame, CancelFrame, etc.) for pipeline lifecycle.
        await self.push_frame(frame, direction)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  VoicePipelineManager
# ─────────────────────────────────────────────────────────────────────────────
#
# NEW in this version:
#   Feature 1 (Interrupt Fix): interrupt() now also drains the output_queue
#     so stale TTS audio frames don't get sent to browser after interruption.
#   Feature 4 (Barge-in): exposes set_barge_in_mode() which delegates to VAD.
#

class VoicePipelineManager:
    """
    Manages the Pipecat pipeline for a single WebSocket connection.
    Each WebSocket gets its own isolated instance with its own memory.
    """

    def __init__(
        self,
        session_id:  Optional[str] = None,
        trace_store  = None,
    ):
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self.thread_id:    str           = str(uuid.uuid4())
        self.session_id:   str           = session_id or str(uuid.uuid4())
        logger.info(
            f"VoicePipelineManager: thread_id={self.thread_id[:8]}… "
            f"session_id={self.session_id[:8]}…"
        )

        self._vad  = VADProcessor()
        # STT transport is flag-selectable. Default (flag off) = batch HTTP,
        # byte-identical to before. STT_STREAMING=true = streaming WebSocket
        # (audio streamed to Sarvam as the user speaks; falls back to batch
        # per-utterance on any failure — see SarvamSTTStreamingService).
        if STT_STREAMING:
            self._stt = SarvamSTTStreamingService(api_key=SARVAM_API_KEY)
            logger.info("STT: streaming WebSocket mode ENABLED (STT_STREAMING=true)")
        else:
            self._stt = SarvamSTTService(api_key=SARVAM_API_KEY)
        # TTS transport is flag-selectable. Default (flag off) = batch HTTP,
        # byte-identical to before. TTS_STREAMING=true = streaming WebSocket.
        if TTS_STREAMING:
            self._tts = SarvamTTSStreamingService(api_key=SARVAM_API_KEY)
            logger.info("TTS: streaming WebSocket mode ENABLED (TTS_STREAMING=true)")
        else:
            self._tts = SarvamTTSService(api_key=SARVAM_API_KEY)
        self._llm  = GroqLangGraphProcessor(
            thread_id   = self.thread_id,
            tts_service = self._tts,
            session_id  = self.session_id,
            trace_store = trace_store,
        )
        self._sink = OutputSink(output_queue=self.output_queue)

        self._pipeline = Pipeline([
            self._vad,
            self._stt,
            self._llm,
            self._tts,
            self._sink,
        ])

        self._task   = PipelineTask(
            self._pipeline,
            params=PipelineParams(allow_interruptions=False),
            enable_rtvi=False,
        )
        self._runner      = PipelineRunner()
        self._runner_coro = None

    def update_sample_rate(self, rate: int):
        """Call once the browser sends its init metadata with the real sample rate."""
        self._vad.update_sample_rate(rate)

    def set_barge_in_mode(self, enabled: bool):
        """
        Feature 4: Enable/disable barge-in mode on the VAD.
        Call with True when AI starts speaking, False when AI stops.
        This lets the VAD detect user speech during AI playback and
        emit BargeInDetectedFrame to trigger immediate interruption.
        """
        self._vad.set_barge_in_mode(enabled)

    async def start(self):
        """Launch the pipeline runner as a background asyncio Task."""
        self._runner_coro = asyncio.create_task(
            self._runner.run(self._task),
            name="pipecat-pipeline-runner",
        )
        logger.info("VoicePipelineManager: pipeline started")

    async def trigger_greeting(self):
        """
        Trigger the agent's opening greeting without waiting for the customer
        to speak first. Bypasses VAD/STT and calls _generate() directly with
        a sentinel input that the system prompt's OPENING rule handles.
        """
        await asyncio.sleep(0.8)   # brief pause so browser is ready to receive audio
        logger.info("VoicePipelineManager: triggering agent opening greeting")
        asyncio.create_task(
            self._llm._generate("__greeting__"),
            name="agent-greeting",
        )

    async def push_audio(self, pcm_bytes: bytes, sample_rate: int = 48000):
        """Inject a raw PCM audio chunk from the browser into the pipeline."""
        frame = InputAudioRawFrame(
            audio=pcm_bytes,
            sample_rate=sample_rate,
            num_channels=1,
        )
        await self._task.queue_frame(frame)

    async def interrupt(self):
        """
        Barge-in / manual-stop handler. Cancels the CURRENT turn only and drains
        stale output — WITHOUT tearing down the pipeline, so the agent keeps
        listening and can respond to the next utterance on the same connection.

        WHY NOT self._task.cancel(): PipelineTask.cancel() permanently finishes
        the task/runner (run() short-circuits on has_finished() and can never be
        re-entered), and nothing recreates it — start() is called exactly once.
        Using it here made the agent go deaf+mute after the FIRST barge-in for the
        rest of the call. Turn-level cancel_turn() is the correct scope: it cancels
        in-flight TTS HTTP tasks, resolves pending futures, and stops the delivery
        loop, leaving _task/_runner alive. (The pipeline runs with
        allow_interruptions=False, so Pipecat's own interruption path is off and
        this manual per-turn cancel is the intended mechanism.)
        """
        # Cancel the in-flight turn (TTS tasks + delivery loop) — non-destructive.
        await self._tts.cancel_turn()

        # Flag the just-cancelled turn as awaiting resolution: if STT then
        # reports the interrupting audio wasn't real speech (NoSpeechDetectedFrame),
        # GroqLangGraphProcessor resumes it instead of leaving the reply
        # silently abandoned. If real speech follows, it's discarded normally.
        self._llm.mark_turn_cancelled()

        # Drain all pending output frames so stale audio/status isn't sent
        drained = 0
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break

        logger.info(f"VoicePipelineManager: interrupted current turn (drained {drained} stale frames)")

    async def stop(self):
        """Gracefully shut down the pipeline when the WebSocket disconnects."""
        try:
            await self._task.queue_frame(EndFrame())
            if self._runner_coro and not self._runner_coro.done():
                await asyncio.wait_for(self._runner_coro, timeout=3.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        # Free this connection's checkpointer state so the in-RAM MemorySaver
        # doesn't grow unbounded across connections (Bug #7).
        await self._llm.cleanup_threads()
        logger.info("VoicePipelineManager: pipeline stopped")

    def clear_memory(self):
        """Reset conversation memory by assigning a new thread_id."""
        self._llm.reset_thread()
        self.thread_id = self._llm._thread_id
        logger.info("VoicePipelineManager: memory cleared (new thread_id)")
