import os
from dotenv import load_dotenv

load_dotenv()

# ── Existing ──────────────────────────────────────────────────────────────────
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

# ── Phase 1: Qdrant Cloud ─────────────────────────────────────────────────────
QDRANT_URL     = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# ── Phase 4: LangSmith observability ─────────────────────────────────────────
LANGSMITH_API_KEY  = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT  = os.getenv("LANGSMITH_PROJECT", "DreamSupport")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_TRACING  = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

# ── Groq LLM models (single source of truth) ───────────────────────────────────
# IMPORTANT: llama-3.1-8b-instant and llama-3.3-70b-versatile are being
# DECOMMISSIONED by Groq on 2026-08-16. Both roles now use GPT-OSS-20B, the
# recommended free-tier replacement: 1000 tok/s (fastest on Groq), cheapest,
# 131K context window. To migrate models in future, change only these two lines.
#
#   VOICE_LLM_MODEL — the live voice agent (latency-critical, every turn)
#   DREAM_LLM_MODEL — the offline Dream Engine (quality over speed)
#
# Other free-tier options if you want to experiment:
#   openai/gpt-oss-120b  — deeper reasoning, slower (500 tok/s), pricier
#   qwen/qwen3-32b       — preview tier
VOICE_LLM_MODEL            = os.getenv("VOICE_LLM_MODEL", "openai/gpt-oss-20b")

# ── Phase 5: Dream Engine ──────────────────────────────────────────────────────
# GPT-OSS-20B replaces the deprecated llama-3.3-70b-versatile. It runs offline
# so latency doesn't matter; its large free-tier budget and 131K context are
# more than enough for evaluation/clustering work, and sharing one model with
# the voice agent keeps token-budget accounting simple.
DREAM_LLM_MODEL            = os.getenv("DREAM_LLM_MODEL", "openai/gpt-oss-20b")
# Idle threshold: how long the system must be quiet before dreaming starts.
# 30s was too eager — it dreamed constantly between calls. 300s (5 min) means
# the engine only runs when the system is genuinely idle.
DREAM_IDLE_THRESHOLD_SECS  = int(os.getenv("DREAM_IDLE_THRESHOLD_SECS", "300"))
# Interval between cycles. 60s burned through budget fast; 300s paces it out.
DREAM_CYCLE_INTERVAL_SECS  = int(os.getenv("DREAM_CYCLE_INTERVAL_SECS", "300"))
# HARD daily token cap for the Dream Engine. The free Groq tier gives ~100K
# tokens/day shared with the voice agent; this reserves the rest for live calls.
# Dreaming stops for the day (or backs off) once it spends this many tokens.
# 50000 = 50% of the free-tier daily budget. Lower this to protect voice further.
DREAM_DAILY_TOKEN_BUDGET   = int(os.getenv("DREAM_DAILY_TOKEN_BUDGET", "50000"))

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_DIM      = int(os.getenv("EMBEDDING_DIM", "384"))
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")   # "local" | "openai"
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")            # only needed if EMBEDDING_PROVIDER=openai
