"""
trace_store.py — Execution trace recording  (Phase 3)
======================================================

Every voice conversation turn produces a TurnTrace that is persisted to the
Qdrant execution_traces collection.  These traces are the raw material that
the Dream Engine analyzes during idle time to improve the bot.

Public API
----------
  ExecutionTraceStore.begin_session(session_id, customer_id, language)
  ExecutionTraceStore.record_turn(session_id, turn: TurnTrace)
  ExecutionTraceStore.end_session(session_id)
  ExecutionTraceStore.get_session_traces(session_id) -> list[TurnTrace]
  ExecutionTraceStore.get_unprocessed_sessions(limit) -> list[str]
  ExecutionTraceStore.update_eval_score(session_id, turn_index, score)

Storage layout
--------------
  Qdrant collection : execution_traces
  Vector           : dummy [0.0]*768  (payload-only collection)
  Point ID         : "{session_id}:{turn_index}"  (deterministic, idempotent)
  Payload fields   : all TurnTrace fields + session metadata
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from app.store import CUSTOMER_PROFILES, EXECUTION_TRACES, QdrantStore


# ---------------------------------------------------------------------------
# TurnTrace dataclass
# ---------------------------------------------------------------------------

@dataclass
class TurnTrace:
    """
    One turn of a customer support conversation.

    All fields except eval_score and customer_feedback are populated at
    recording time.  eval_score is filled later by FailureAnalysisCycle.
    """
    session_id:        str
    turn_index:        int
    user_input:        str
    detected_language: str
    retrieved_docs:    list[dict]          # RAG chunks used (doc_id, content[:200], score)
    tool_calls:        list[dict]          # web_search calls made this turn
    ai_response:       str
    latency_ms:        int
    emotion_hint:      str                 # "neutral" | "hesitant" | "agitated"
    created_at:        str                 # ISO-8601 UTC
    eval_score:        Optional[float] = None   # filled by Dream Cycle
    customer_feedback: Optional[str]  = None   # filled if customer rates the turn
    dream_processed:   bool           = False  # True after Dream Cycle has analyzed it
    # Session-level conversion outcome, stamped on every turn at end_session:
    # "converted" | "interested" | "info_only" | "lost" | None (unknown/empty).
    # Deterministically inferred from the transcript (agent close + customer
    # agreement / [END_CALL]). The real reward signal for the Dream Engine.
    session_outcome:   Optional[str]  = None


# ---------------------------------------------------------------------------
# Conversion-outcome classifier  (deterministic, no LLM)
# ---------------------------------------------------------------------------
import re as _re

# Agent phrases that indicate a CLOSE attempt (asking to register interest).
_CLOSE_ATTEMPT = _re.compile(
    r"register your interest|note your interest|shall i go ahead|"
    r"go ahead and (register|note|set)|"
    r"our team will call you back|complete the setup|"
    r"रुचि दर्ज|आपकी जानकारी नोट|आमची टीम.*कॉल|नोंदवतो",
    _re.IGNORECASE,
)

# Customer AGREEMENT to a close (short affirmatives + common Hindi/Marathi yes).
_AGREE = _re.compile(
    r"\b(yes|yeah|yep|sure|okay do it|go ahead|please do|sounds good|"
    r"register me|sign me up|let'?s do it|i'?m in)\b|"
    r"हाँ|हां|कर दो|कर दीजिए|ठीक है कर|हो कर|नोंदवा|करा",
    _re.IGNORECASE,
)

# Customer INTEREST (leaning in, but not an explicit yes-to-close).
_INTEREST = _re.compile(
    r"\b(interested|tell me more|how do i|how much|sounds good|"
    r"i want|i'?d like|can i get|which plan)\b|"
    r"चाहिए|कितने का|कैसे|हवा आहे|किती",
    _re.IGNORECASE,
)


def classify_session_outcome(turns: list["TurnTrace"]) -> str:
    """
    Infer a conversion outcome for a whole session from its transcript.

    Returns one of: "converted" | "interested" | "info_only" | "lost".
    Deterministic (no LLM, no tokens):
      - converted: the agent attempted a close AND the customer agreed on the
        NEXT turn (or the agent's closing/goodbye turn followed agreement).
      - interested: the customer showed buying signals (asked prices/plans, said
        they want something) but there was no confirmed close.
      - info_only: a short, purely informational exchange with no buying signal.
      - lost: multiple turns with a buying signal present but never converted,
        i.e. the sale slipped away.
    Approximate by design — a real, free signal that beats per-turn answer scores.
    """
    if not turns:
        return "info_only"

    agent_closed = False
    customer_agreed_after_close = False
    any_interest = False

    for i, t in enumerate(turns):
        ai = t.ai_response or ""
        user = t.user_input or ""
        if _INTEREST.search(user):
            any_interest = True
        if _CLOSE_ATTEMPT.search(ai):
            agent_closed = True
            # Look at the NEXT user turn for agreement.
            if i + 1 < len(turns) and _AGREE.search(turns[i + 1].user_input or ""):
                customer_agreed_after_close = True

    if agent_closed and customer_agreed_after_close:
        return "converted"
    if any_interest or agent_closed:
        # Buying signal present but no confirmed yes. If the agent even reached a
        # close and the customer didn't agree, count it as a lost opportunity;
        # otherwise they were interested but the call ended before closing.
        return "lost" if agent_closed else "interested"
    return "info_only"


def _turn_point_id(session_id: str, turn_index: int) -> str:
    """
    Deterministic Qdrant point ID for a turn.
    Using a UUID derived from session_id + turn_index makes upserts idempotent
    (re-recording the same turn overwrites the existing point, no duplicates).
    """
    seed = f"{session_id}:{turn_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


# ---------------------------------------------------------------------------
# ExecutionTraceStore
# ---------------------------------------------------------------------------

class ExecutionTraceStore:
    """
    Persists turn-by-turn execution traces to Qdrant.

    One instance per app lifetime — pass the shared QdrantStore.
    All methods are async and safe to call from any coroutine.
    """

    def __init__(self, store: QdrantStore) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def begin_session(
        self,
        session_id: str,
        customer_id: str = "anonymous",
        language: str    = "unknown",
    ) -> None:
        """
        Called at WebSocket connect.  Creates or updates the customer_profiles
        record for this customer so we track last_seen_at.

        For anonymous customers we use a synthetic profile keyed by session_id
        so that the Dream Cycle can still link traces to a "customer".
        """
        effective_customer_id = customer_id if customer_id != "anonymous" else session_id

        profile_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"customer:{effective_customer_id}"))
        now        = _utc_now()

        await self._store.upsert(
            CUSTOMER_PROFILES,
            [{
                "id":     profile_id,
                "vector": self._store.dummy_vector(),
                "payload": {
                    "customer_id":         effective_customer_id,
                    "name":                customer_id,
                    "language_preference": language,
                    "last_seen_at":        now,
                    "past_issue_summary":  "",
                    "preferences":         {},
                },
            }],
        )
        logger.info(
            f"TraceStore: session started — session_id={session_id[:8]}… "
            f"customer_id={effective_customer_id[:8]}…"
        )

    async def end_session(self, session_id: str) -> None:
        """
        Called at WebSocket disconnect. Infers the session's CONVERSION OUTCOME
        from the transcript and stamps it on every turn of the session, so the
        Dream Engine can learn from what actually drove conversions (not just
        per-turn answer quality). Non-fatal: any failure just logs.
        """
        try:
            turns = await self.get_session_traces(session_id)
            outcome = classify_session_outcome(turns)
            # Stamp the label on each turn (turns are the only per-session record).
            for t in turns:
                try:
                    await self._store.update_payload(
                        EXECUTION_TRACES,
                        _turn_point_id(session_id, t.turn_index),
                        {"session_outcome": outcome},
                    )
                except Exception as exc:
                    logger.warning(f"end_session: outcome stamp failed (non-fatal): {exc}")
            logger.info(
                f"TraceStore: session ended — session_id={session_id[:8]}… "
                f"outcome={outcome} ({len(turns)} turns)"
            )
        except Exception as exc:
            logger.warning(f"end_session: outcome classification failed (non-fatal): {exc}")
            logger.info(f"TraceStore: session ended — session_id={session_id[:8]}…")

    # ------------------------------------------------------------------
    # Turn recording
    # ------------------------------------------------------------------

    async def record_turn(self, turn: TurnTrace) -> None:
        """
        Upsert a single turn into execution_traces.

        Idempotent: recording the same (session_id, turn_index) pair twice
        overwrites the first record (safe to retry on transient errors).
        """
        point_id = _turn_point_id(turn.session_id, turn.turn_index)
        payload  = asdict(turn)

        await self._store.upsert(
            EXECUTION_TRACES,
            [{
                "id":      point_id,
                "vector":  self._store.dummy_vector(),
                "payload": payload,
            }],
        )
        logger.debug(
            f"TraceStore: recorded turn {turn.turn_index} "
            f"session={turn.session_id[:8]}… "
            f"latency={turn.latency_ms}ms "
            f"rag={len(turn.retrieved_docs)} chunks"
        )

    # ------------------------------------------------------------------
    # Reads (used by Dream Cycle)
    # ------------------------------------------------------------------

    async def get_session_traces(self, session_id: str) -> list[TurnTrace]:
        """
        Return all turns for a session, sorted by turn_index ascending.
        """
        records, _ = await self._store.scroll(
            collection=EXECUTION_TRACES,
            filter=self._store.filter_eq("session_id", session_id),
            limit=500,
        )
        turns = [_record_to_turn(r["payload"]) for r in records]
        turns.sort(key=lambda t: t.turn_index)
        return turns

    async def get_unprocessed_sessions(self, limit: int = 50) -> list[str]:
        """
        Return session IDs that have at least one turn with dream_processed=False.
        Used by FailureAnalysisCycle to find work to do.

        Returns distinct session IDs (deduped).
        """
        records, _ = await self._store.scroll(
            collection=EXECUTION_TRACES,
            filter=self._store.filter_eq("dream_processed", False),
            limit=limit * 5,  # over-fetch because many turns share one session
        )
        seen:    set[str] = set()
        session_ids: list[str] = []
        for r in records:
            sid = r["payload"].get("session_id", "")
            if sid and sid not in seen:
                seen.add(sid)
                session_ids.append(sid)
                if len(session_ids) >= limit:
                    break
        return session_ids

    async def get_unprocessed_turns(self, limit: int = 50) -> list[TurnTrace]:
        """
        Return individual turns with dream_processed=False (batch for analysis).
        Used by FailureAnalysisCycle to score turns one-by-one.
        """
        records, _ = await self._store.scroll(
            collection=EXECUTION_TRACES,
            filter=self._store.filter_eq("dream_processed", False),
            limit=limit,
        )
        return [_record_to_turn(r["payload"]) for r in records]

    # ------------------------------------------------------------------
    # Updates (written by Dream Cycle)
    # ------------------------------------------------------------------

    async def update_eval_score(
        self,
        session_id: str,
        turn_index: int,
        score:      float,
        issues:     list[str] | None = None,
        dimensions: dict | None = None,
    ) -> None:
        """
        Write the LLM-judged quality score back into the trace.
        Also marks dream_processed=True so this turn isn't re-analyzed.
        Called by FailureAnalysisCycle after scoring each turn.

        dimensions: optional per-axis scores (correctness/helpfulness/
        sales_progress) so downstream cycles and trend analysis can see WHY a
        turn scored low — e.g. correct but no sales progress.
        """
        point_id = _turn_point_id(session_id, turn_index)
        patch: dict = {
            "eval_score":       round(score, 3),
            "dream_processed":  True,
        }
        if issues is not None:
            patch["issues"] = issues
        if dimensions:
            patch["eval_dimensions"] = dimensions

        await self._store.update_payload(EXECUTION_TRACES, point_id, patch)
        logger.debug(
            f"TraceStore: eval_score={score:.1f} written for "
            f"session={session_id[:8]}… turn={turn_index}"
        )

    async def mark_dream_processed(
        self,
        session_id: str,
        turn_index: int,
    ) -> None:
        """Mark a turn as processed without setting an eval_score."""
        point_id = _turn_point_id(session_id, turn_index)
        await self._store.update_payload(
            EXECUTION_TRACES,
            point_id,
            {"dream_processed": True},
        )

    async def update_customer_feedback(
        self,
        session_id: str,
        turn_index: int,
        feedback:   str,
    ) -> None:
        """Store optional customer feedback (e.g. thumbs up/down) on a turn."""
        point_id = _turn_point_id(session_id, turn_index)
        await self._store.update_payload(
            EXECUTION_TRACES,
            point_id,
            {"customer_feedback": feedback},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _record_to_turn(payload: dict) -> TurnTrace:
    """Reconstruct a TurnTrace from a Qdrant payload dict."""
    return TurnTrace(
        session_id        = payload.get("session_id",        ""),
        turn_index        = int(payload.get("turn_index",    0)),
        user_input        = payload.get("user_input",        ""),
        detected_language = payload.get("detected_language", "unknown"),
        retrieved_docs    = payload.get("retrieved_docs",    []),
        tool_calls        = payload.get("tool_calls",        []),
        ai_response       = payload.get("ai_response",       ""),
        latency_ms        = int(payload.get("latency_ms",    0)),
        emotion_hint      = payload.get("emotion_hint",      "neutral"),
        created_at        = payload.get("created_at",        ""),
        eval_score        = payload.get("eval_score"),
        customer_feedback = payload.get("customer_feedback"),
        dream_processed   = bool(payload.get("dream_processed", False)),
    )
