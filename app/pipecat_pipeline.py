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
import os
import tempfile
import torch
import uuid
import wave
from typing import List

import httpx
from loguru import logger

# ─────────────────────────────────────────────────────────────────────────────
# Pipecat Core Imports
# ─────────────────────────────────────────────────────────────────────────────
from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    Frame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

from app.config import SARVAM_API_KEY

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


class LLMTurnDoneFrame(Frame):
    """
    Emitted by GroqLangGraphProcessor after all TextFrames for a turn have
    been pushed. Travels through the pipeline IN ORDER so SarvamTTSService
    receives it only after every TextFrame for this turn has been processed.
    This replaces the external flush() call which had a race condition:
    flush() could be called before the last TextFrame reached process_frame.
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
    SPEECH_THRESHOLD     = 0.5   # Silero probability above this = speech
    MIN_SPEECH_CHUNKS    = 3     # ~0.25 s of confirmed speech before buffering
    SILENCE_CHUNKS_NEEDED = 8   # ~0.67 s of quiet = utterance ended (normal mode)
    SILENCE_CHUNKS_BARGEIN = 3  # Feature 4: faster end-of-speech during barge-in
    MAX_BUFFER_CHUNKS    = 80   # ~6.7 s safety cap

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
        """Downsample from browser rate to 16 kHz using integer decimation."""
        if self._browser_rate == self.TARGET_SAMPLE_RATE:
            return pcm_bytes
        ratio = self._browser_rate / self.TARGET_SAMPLE_RATE
        samples = array.array('h', pcm_bytes[:len(pcm_bytes) & ~1])
        out_len = int(len(samples) / ratio)
        out = array.array('h', (samples[int(i * ratio)] for i in range(out_len)))
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

    async def _process_audio_chunk(self, raw_pcm: bytes):
        """Run one chunk of browser audio through the Silero VAD state machine."""
        pcm_16k = self._resample(raw_pcm)
        prob    = self._speech_prob(pcm_16k)

        if prob >= self.SPEECH_THRESHOLD:
            # ── SPEECH chunk ─────────────────────────────────────────────────
            self._silence_chunk_count = 0
            self._audio_buffer.append(pcm_16k)

            if not self._is_speech_active:
                self._speech_chunks_seen += 1
                if self._speech_chunks_seen >= self.MIN_SPEECH_CHUNKS:
                    self._is_speech_active = True
                    logger.debug(f"VAD: speech STARTED (prob={prob:.2f}, barge_in={self._barge_in_mode})")

                    # Feature 4: Emit barge-in signal the moment speech is confirmed
                    # while AI is speaking. main.py will interrupt the AI immediately.
                    if self._barge_in_mode and not self._barge_in_signaled:
                        self._barge_in_signaled = True
                        await self.push_frame(BargeInDetectedFrame())

            # Feature 10: track energy during active speech
            if self._is_speech_active:
                self._track_energy(pcm_16k)

        else:
            # ── SILENCE chunk ─────────────────────────────────────────────────
            if self._is_speech_active:
                self._silence_chunk_count += 1
                self._audio_buffer.append(pcm_16k)

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
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="pipecat_utt_")
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(frame.audio_bytes)

            with open(tmp_path, "rb") as f:
                resp = await self._http.post(
                    self.SARVAM_ASR_URL,
                    headers={"api-subscription-key": self._api_key},
                    files={"file": ("audio.wav", f, "audio/wav")},
                    data={
                        "model": "saarika:v2.5",
                        # Always "unknown" — let Sarvam auto-detect every turn.
                        # This is the only way to correctly handle mid-conversation
                        # language switches (English → Hindi → English → Marathi…).
                        "language_code": "unknown",
                        "with_disfluencies": "false",
                    },
                )

            if resp.status_code != 200:
                logger.error(f"STT HTTP {resp.status_code}: {resp.text[:200]}")
                return

            resp_json   = resp.json()
            transcript  = resp_json.get("transcript", "").strip()
            logger.info(f"STT transcript: {transcript!r}")

            # ── Feature 7: Language detection ──────────────────────────────
            # Sarvam returns the detected language code in the response.
            # We normalise it, then apply a romanized-language correction pass:
            # Sarvam's auto-detect sometimes returns "en" for romanized Indian
            # speech (e.g. Hinglish, romanized Marathi, etc.) because the text
            # looks Latin-script. We scan the transcript for known function words
            # of each Indian language and override when we get ≥2 hits.
            raw_lang = resp_json.get("language_code", "en")
            detected = self.LANG_NORMALIZE.get(raw_lang, raw_lang)

            if detected == "en-IN" and transcript:
                words = set(transcript.lower().replace(",", " ").replace(".", " ").split())
                best_lang  = None
                best_count = 0
                for lang_code, markers in self.ROMANIZED_MARKERS.items():
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
            await self.push_frame(LanguageDetectedFrame(language_code=detected))

            # ── Feature 10: Emotion hint from confidence ────────────────────
            # Sarvam may return a confidence score. Low confidence suggests
            # the user was hesitant, unclear, or mumbling.
            confidence = float(resp_json.get("confidence", 1.0))
            if confidence < 0.6:
                logger.info(f"STT: low confidence ({confidence:.2f}) → hesitant emotion hint")
                await self.push_frame(EmotionHintFrame(hint="hesitant"))

            # ── Noise / filler filter ──────────────────────────────────────
            cleaned = transcript.lower().rstrip(".,!? ")
            if not cleaned or len(cleaned) <= 2 or cleaned in self.FILLER_WORDS:
                logger.debug(f"STT: filtered noise/filler {transcript!r}")
                return

            # ── Emit frames downstream ─────────────────────────────────────
            # ORDERING: push user display frame FIRST, then the transcription.
            # TranscriptionFrame triggers a long-running LLM call which blocks
            # the pipeline. Pushing the display frame first ensures the user's
            # text bubble appears in the browser BEFORE the AI response audio.
            await self.push_frame(
                TranscriptDisplayFrame(text=transcript, speaker="user")
            )
            await self.push_frame(
                TranscriptionFrame(text=transcript, user_id="user", timestamp="")
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

    def __init__(self, thread_id: str, tts_service=None, **kwargs):
        super().__init__(**kwargs)
        self._thread_id    = thread_id
        self._emotion_hint = "neutral"   # Feature 10: updated by EmotionHintFrame
        self._language     = "en-IN"     # updated by LanguageDetectedFrame each turn
        self._tts          = tts_service # reference to SarvamTTSService for flush()
        logger.info(f"GroqLangGraphProcessor: thread_id={thread_id}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            await self._generate(frame.text)

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

    async def _generate(self, user_text: str):
        """
        Stream user text through LangGraph + Groq, emitting one TextFrame
        per sentence for immediate TTS synthesis.

        Flow:
          1. Emit AIThinkingFrame(True)  → browser shows "AI is thinking…"
          2. Call stream_agent() which streams sentences from the LLM
          3. On first sentence: emit AIThinkingFrame(False) to hide thinking dots
          4. Push each sentence as TextFrame → TTS synthesizes immediately
          5. In `finally`: always emit full TranscriptDisplayFrame for chat log
             — even if _save_turn() inside stream_agent raises an exception.
          6. Reset emotion_hint to neutral
        """
        from app.langgraph_flow import stream_agent

        logger.info(f"LangGraph streaming: thread={self._thread_id[:8]}… input={user_text!r}")

        # Feature 6: signal to browser that we're thinking
        await self.push_frame(AIThinkingFrame(thinking=True))

        full_text      = ""
        first_sentence = True

        try:
            async for sentence in stream_agent(user_text, self._thread_id, self._emotion_hint, self._language):
                if not sentence.strip():
                    continue

                if first_sentence:
                    # Feature 6: hide thinking dots as soon as we have the first sentence
                    await self.push_frame(AIThinkingFrame(thinking=False))
                    first_sentence = False

                full_text += sentence.strip() + " "
                logger.info(f"LLM→TTS: pushing TextFrame → {sentence.strip()!r}")
                # Feature 3: push each sentence to TTS immediately
                await self.push_frame(TextFrame(text=sentence.strip()))

        except Exception as e:
            logger.error(f"LangGraph streaming error: {e}")

        finally:
            # Always hide the thinking indicator (even if an error occurred)
            if first_sentence:
                await self.push_frame(AIThinkingFrame(thinking=False))

            # Always show the AI text in the chat, even if state-save failed.
            # This was the root cause of missing AI text bubbles: an exception
            # from _save_turn() inside stream_agent() was caught here and caused
            # an early return before TranscriptDisplayFrame was pushed.
            if full_text.strip():
                await self.push_frame(TranscriptDisplayFrame(text=full_text.strip(), speaker="ai"))

        # Reset emotion hint after use — next turn starts neutral
        self._emotion_hint = "neutral"

        # Push the sentinel THROUGH the pipeline so it arrives at SarvamTTSService
        # only AFTER all the TextFrames from this turn have been processed.
        # This fixes the race where the old flush() call happened before the last
        # TextFrame reached SarvamTTSService.process_frame.
        logger.info("LLM: pushing LLMTurnDoneFrame sentinel downstream")
        await self.push_frame(LLMTurnDoneFrame())

    def reset_thread(self):
        """Start a new conversation by assigning a new thread_id."""
        old = self._thread_id
        self._thread_id = str(uuid.uuid4())
        logger.info(f"LangGraph: thread reset {old[:8]}→{self._thread_id[:8]}")


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
        # In-order delivery: list of futures in arrival order
        self._pending: list    = []
        self._delivery_task    = None
        # Lazily created in process_frame (needs a running loop)
        self._llm_done: asyncio.Event | None = None
        # Track background tasks so interrupt() can cancel them
        self._inflight_tasks: list = []

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
            tts_text = self._truncate(text)
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
                    "speaker":              "anushka",
                    "model":                "bulbul:v2",
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
                    await self.push_frame(AIStatusFrame(ai_speaking=True))

                await self.push_frame(AIAudioFrame(audio_bytes=wav_bytes))

        except asyncio.CancelledError:
            logger.warning("TTS delivery: cancelled mid-delivery")
            pass  # interrupted — finally block handles cleanup
        except Exception as e:
            logger.error(f"TTS delivery error: {e}")
        finally:
            logger.info(f"TTS delivery: loop exiting — delivered {chunk_n} chunks, tts_active={self._tts_active}")
            if self._tts_active:
                await self.push_frame(AIStatusFrame(ai_speaking=False))
            self._tts_active = False
            # Clear only this turn's pending list — NOT self._pending, which
            # may have already been replaced by a new turn's list object.
            my_pending.clear()

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
        if self._llm_done is not None:
            self._llm_done.clear()

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

    def __init__(self):
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self.thread_id: str = str(uuid.uuid4())
        logger.info(f"VoicePipelineManager: new session thread_id={self.thread_id[:8]}…")

        self._vad  = VADProcessor()
        self._stt  = SarvamSTTService(api_key=SARVAM_API_KEY)
        self._tts  = SarvamTTSService(api_key=SARVAM_API_KEY)
        self._llm  = GroqLangGraphProcessor(thread_id=self.thread_id, tts_service=self._tts)
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

    async def push_audio(self, pcm_bytes: bytes, sample_rate: int = 48000):
        """Inject a raw PCM audio chunk from the browser into the pipeline."""
        frame = AudioRawFrame(
            audio=pcm_bytes,
            sample_rate=sample_rate,
            num_channels=1,
        )
        await self._task.queue_frame(frame)

    async def interrupt(self):
        """
        Feature 1 (Interrupt Fix): Cancel current pipeline processing AND drain
        the output_queue so no stale TTS audio is sent to the browser afterwards.

        Also cancels any in-flight TTS background tasks so they don't push
        audio frames after the interrupt (barge-in / manual stop).
        """
        # Cancel in-flight TTS tasks and stop the delivery loop first,
        # so no new frames are pushed to the queue after we drain it.
        await self._tts.cancel_turn()

        await self._task.cancel()

        # Drain all pending output frames so stale audio/status isn't sent
        drained = 0
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break

        logger.info(f"VoicePipelineManager: interrupted (drained {drained} stale frames)")

    async def stop(self):
        """Gracefully shut down the pipeline when the WebSocket disconnects."""
        try:
            await self._task.queue_frame(EndFrame())
            if self._runner_coro and not self._runner_coro.done():
                await asyncio.wait_for(self._runner_coro, timeout=3.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        logger.info("VoicePipelineManager: pipeline stopped")

    def clear_memory(self):
        """Reset conversation memory by assigning a new thread_id."""
        self._llm.reset_thread()
        self.thread_id = self._llm._thread_id
        logger.info("VoicePipelineManager: memory cleared (new thread_id)")
