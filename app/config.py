import os
from dotenv import load_dotenv

load_dotenv()

# ── Existing ──────────────────────────────────────────────────────────────────
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

# Gemini (voice LLM) — used via its OpenAI-compatible endpoint. Kept on a
# SEPARATE provider from Groq (dream) so the two draw from independent free-tier
# pools: dreaming can never exhaust the live voice quota, and vice-versa.
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
)

# ── Phase 1: Qdrant Cloud ─────────────────────────────────────────────────────
QDRANT_URL     = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# ── Phase 4: LangSmith observability ─────────────────────────────────────────
LANGSMITH_API_KEY  = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT  = os.getenv("LANGSMITH_PROJECT", "DreamSupport")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_TRACING  = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

# ── LLM models (single source of truth) ────────────────────────────────────────
# The voice agent and the Dream Engine run on DIFFERENT providers, chosen for
# opposite needs. Migrate either by editing these lines (or the .env) only —
# never hardcode a model id in app code.
#
#   VOICE_LLM_MODEL — live voice agent (latency-critical). Gemini Flash-Lite via
#     its OpenAI-compatible endpoint: fast first token (no reasoning tax),
#     strong Indian-language + native-script handling, streaming + tools.
#   DREAM_LLM_MODEL — offline Dream Engine (quality over speed). Groq
#     gpt-oss-120b: a reasoning model, which is an asset for offline
#     evaluation/clustering/self-critique where latency doesn't matter.
VOICE_LLM_MODEL            = os.getenv("VOICE_LLM_MODEL", "gemini-3.1-flash-lite")

# ── Phase 5: Dream Engine ──────────────────────────────────────────────────────
DREAM_LLM_MODEL            = os.getenv("DREAM_LLM_MODEL", "openai/gpt-oss-120b")
# Idle threshold: how long the system must be quiet before dreaming starts.
# 30s was too eager — it dreamed constantly between calls. 300s (5 min) means
# the engine only runs when the system is genuinely idle.
DREAM_IDLE_THRESHOLD_SECS  = int(os.getenv("DREAM_IDLE_THRESHOLD_SECS", "300"))
# Interval between cycles. 60s burned through budget fast; 300s paces it out.
DREAM_CYCLE_INTERVAL_SECS  = int(os.getenv("DREAM_CYCLE_INTERVAL_SECS", "300"))
# HARD daily token cap for the Dream Engine. Now that dreaming (Groq) and the
# voice agent (Gemini) are on SEPARATE providers, dreaming no longer shares a
# budget with live calls — so it gets the FULL gpt-oss-120b free-tier day
# (~200K tokens/day). This cap is a self-imposed safety ceiling; Groq's own 429
# is the real backstop. Dreaming stops for the day (or backs off) once it spends
# this many tokens. Lower it only if you want dreaming to do less per day.
DREAM_DAILY_TOKEN_BUDGET   = int(os.getenv("DREAM_DAILY_TOKEN_BUDGET", "200000"))

# ── TTS transport ───────────────────────────────────────────────────────────
# Selects the TTS transport. Default is now the streaming Sarvam TTS WebSocket
# (SarvamTTSStreamingService). The per-sentence batch HTTP client
# (SarvamTTSService) is kept as an instant rollback: set TTS_STREAMING=false to
# restore the exact prior behavior with no code change — no redeploy of logic,
# just an env flip. Remove the batch path only after streaming is verified live.
TTS_STREAMING = os.getenv("TTS_STREAMING", "true").lower() == "true"

# ── STT transport ───────────────────────────────────────────────────────────
# Mirrors TTS_STREAMING. Default false: the batch HTTP path (SarvamSTTService)
# remains the exact prior behavior until the streaming path (SarvamSTTStreamingService)
# is verified against a real live call. When true, audio is streamed to
# Sarvam's STT WebSocket continuously as the user speaks — instead of one
# blocking batch call after silence is detected — so ASR compute overlaps
# their speaking time. Falls back to the batch path automatically per
# utterance on any streaming failure/timeout, so flipping this on can only
# change latency, never reliability. Flip back to false instantly via .env,
# no redeploy, if anything looks wrong live.
STT_STREAMING = os.getenv("STT_STREAMING", "false").lower() == "true"

# ── Sarvam speech models (single source of truth) ─────────────────────────────
# Migrate Sarvam STT/TTS versions from .env only — never hardcode model IDs or
# speakers in app code. Defaults track Sarvam's current recommendations:
#   STT: saaras:v3 (saarika:v2.5 is legacy and being deprecated). `mode` is only
#        valid on saaras:* models — "transcribe" keeps original-language output
#        (vs "translate", which would force English).
#   TTS: bulbul:v3. NOTE speaker names are model-specific — bulbul:v3 does NOT
#        accept the old v2 speakers (e.g. "anushka"). Valid v3 speakers include:
#        simran, priya, neha, ritu, pooja, kavya, shreya, shubh, aditya, rahul…
SARVAM_STT_MODEL   = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
SARVAM_STT_MODE    = os.getenv("SARVAM_STT_MODE", "transcribe")
SARVAM_TTS_MODEL   = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "simran")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_DIM      = int(os.getenv("EMBEDDING_DIM", "384"))
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")   # "local" | "openai"
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")            # only needed if EMBEDDING_PROVIDER=openai
