"""
observability.py — LangSmith tracing + eval client  (Phase 4)
==============================================================

Call init_langsmith() once at app startup (inside the FastAPI lifespan).
After that every function decorated with @traceable is automatically traced
in the LangSmith dashboard — no other changes required.

What gets traced automatically:
  • stream_agent()        — full LLM call with token counts, latency, model
  • _retrieve_context()   — RAG retrieval: query, chunks returned, scores
  • web_search tool calls — surfaced as child spans inside stream_agent

Dashboard URL: https://smith.langchain.com  → project "dreamsupport"

Environment variables required (add to .env):
  LANGSMITH_API_KEY=your_key
  LANGSMITH_PROJECT=dreamsupport   (optional, defaults to "dreamsupport")

If LANGSMITH_API_KEY is not set, init_langsmith() is a safe no-op — the app
starts normally and tracing is simply disabled.
"""

from __future__ import annotations

import os
from loguru import logger

from app.config import (
    LANGSMITH_API_KEY,
    LANGSMITH_ENDPOINT,
    LANGSMITH_PROJECT,
    LANGSMITH_TRACING,
)


def init_langsmith() -> bool:
    """
    Activate LangSmith tracing by setting the environment variables that
    LangChain/LangGraph look for.  Must be called before any @traceable
    function is invoked.

    Reads four env vars (all set in .env):
      LANGSMITH_API_KEY   — required
      LANGSMITH_PROJECT   — project name in the dashboard
      LANGSMITH_ENDPOINT  — API endpoint (default: https://api.smith.langchain.com)
      LANGSMITH_TRACING   — "true" / "false" master switch

    Returns True if tracing was enabled, False otherwise (non-fatal).
    """
    if not LANGSMITH_API_KEY:
        logger.warning(
            "LANGSMITH_API_KEY not set — LangSmith tracing disabled. "
            "Add it to .env to enable full observability."
        )
        return False

    if not LANGSMITH_TRACING:
        logger.info("LANGSMITH_TRACING=false — tracing disabled by config.")
        return False

    # These are the exact variable names LangChain/LangSmith SDK reads.
    os.environ["LANGCHAIN_TRACING_V2"]  = "true"
    os.environ["LANGCHAIN_API_KEY"]     = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"]     = LANGSMITH_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"]    = LANGSMITH_ENDPOINT
    # Also set the native LangSmith vars (newer SDK versions read these directly)
    os.environ["LANGSMITH_API_KEY"]     = LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"]     = LANGSMITH_PROJECT
    os.environ["LANGSMITH_ENDPOINT"]    = LANGSMITH_ENDPOINT
    os.environ["LANGSMITH_TRACING"]     = "true"

    logger.info(
        f"LangSmith tracing enabled ✓  "
        f"project='{LANGSMITH_PROJECT}'  "
        f"endpoint='{LANGSMITH_ENDPOINT}'"
    )
    return True


def get_langsmith_client():
    """
    Return a LangSmith Client instance for programmatic use
    (e.g. submitting eval feedback from Dream Cycle).

    Returns None if langsmith is not installed or API key is missing.
    """
    if not LANGSMITH_API_KEY:
        return None
    try:
        from langsmith import Client
        return Client()
    except ImportError:
        logger.warning("langsmith package not installed — pip install langsmith")
        return None
