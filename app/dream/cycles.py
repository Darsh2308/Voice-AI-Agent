"""
dream/cycles.py — Five Dream Sub-Cycles  (Phase 5)
====================================================

Each cycle is a self-contained class with a single public method:

    await cycle.run(pause_signal: asyncio.Event)

The pause_signal is the DreamEngine's _pause_event.  When it becomes set()
(a customer connected), the cycle saves a Qdrant checkpoint and returns
immediately.  On the next idle window the engine creates a fresh instance
and calls run() again; the cycle loads its checkpoint and resumes from where
it left off.

Sub-cycle summary
─────────────────
  1. FailureAnalysisCycle
     Score every unprocessed conversation turn 1-10 using Groq LLM.
     Write eval_score back into execution_traces.  Turns < 6 are flagged.

  2. RetrievalQualityAnalysisCycle
     For each poor-scoring turn, assess whether the RAG context was relevant.
     Log knowledge gaps and query-reformulation hints into improvement_log.

  3. PromptImprovementCycle
     Cluster failure traces by semantic similarity.  Propose one system-prompt
     addendum per cluster.  Judge with a held-out set.  Store approved addenda.

  4. SyntheticQueryGenCycle
     Generate adversarial + edge-case test conversations from failure patterns.
     Store them as synthetic traces in execution_traces.

  5. MemoryConsolidationCycle
     Housekeeping: stale profiles, duplicate detection, KB hygiene, summary
     refresh for frequent customers.

Common checkpoint format (stored in dream_checkpoints collection):
    {
      "cycle_type": str,          # cycle class name
      "progress":   dict,         # cycle-specific resume state
      "started_at": ISO-8601 str,
      "status":     "running" | "done" | "paused",
    }

All LLM calls use DREAM_LLM_MODEL (openai/gpt-oss-20b by default).
All cloud operations are wrapped in try/except so a single Qdrant or Groq
failure never crashes the dream loop — the cycle just logs and continues.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

from app.config import DREAM_LLM_MODEL
from app.store import (
    CUSTOMER_PROFILES,
    DREAM_CHECKPOINTS,
    EXECUTION_TRACES,
    IMPROVEMENT_LOG,
    KNOWLEDGE_BASE,
    QdrantStore,
)

if TYPE_CHECKING:
    from app.dream.budget import DreamTokenBudget
    from app.knowledge.retriever import RetrievalPipeline
    from app.tracing.trace_store import ExecutionTraceStore, TurnTrace
    from groq import AsyncGroq


# ---------------------------------------------------------------------------
# Helpers shared across cycles
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_id(cycle_type: str) -> str:
    """Deterministic Qdrant point ID for a cycle's checkpoint."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"dream_checkpoint:{cycle_type}"))


async def _save_checkpoint(
    store: QdrantStore,
    cycle_type: str,
    progress: dict,
    status: str = "paused",
) -> None:
    """Upsert the checkpoint for a cycle into dream_checkpoints."""
    try:
        await store.upsert(
            DREAM_CHECKPOINTS,
            [{
                "id":      _checkpoint_id(cycle_type),
                "vector":  store.dummy_vector(),
                "payload": {
                    "cycle_type": cycle_type,
                    "progress":   progress,
                    "started_at": _utc_now(),
                    "status":     status,
                },
            }],
        )
    except Exception as exc:
        logger.warning(f"_save_checkpoint({cycle_type}) failed (non-fatal): {exc}")


async def _load_checkpoint(store: QdrantStore, cycle_type: str) -> dict | None:
    """Load the checkpoint for a cycle.  Returns None if no checkpoint exists."""
    try:
        point = await store.get_point(DREAM_CHECKPOINTS, _checkpoint_id(cycle_type))
        if point and point["payload"].get("status") == "paused":
            return point["payload"].get("progress", {})
    except Exception as exc:
        logger.warning(f"_load_checkpoint({cycle_type}) failed (non-fatal): {exc}")
    return None


async def _log_improvement(
    store: QdrantStore,
    category: str,
    description: str,
    before_metric: float | None = None,
    after_metric: float | None = None,
    extra: dict | None = None,
) -> None:
    """
    Write one entry to the improvement_log collection.

    For non-semantic categories (knowledge_gap, retrieval, prompt) a dummy
    vector is used; if sentence-transformers is available we embed the
    description for richer future search.
    """
    vector = store.dummy_vector()
    try:
        from app.knowledge.embedder import embed_text
        vector = await embed_text(description)
    except Exception:
        pass

    payload = {
        "category":     category,
        "improvement_desc": description,
        "before_metric": before_metric,
        "after_metric":  after_metric,
        "applied_at":    _utc_now(),
        "cycle_id":      str(uuid.uuid4()),
    }
    if extra:
        payload.update(extra)

    try:
        await store.upsert(
            IMPROVEMENT_LOG,
            [{
                "id":      store.new_id(),
                "vector":  vector,
                "payload": payload,
            }],
        )
    except Exception as exc:
        logger.warning(f"_log_improvement failed (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# Base cycle class
# ---------------------------------------------------------------------------

# These two are CONTROL-FLOW signals that must abort the whole dream run, not
# per-item errors. They inherit from BaseException (like CancelledError) so the
# many `except Exception` blocks inside cycles do NOT swallow them — only the
# DreamEngine's explicit handlers catch them and trigger the long back-off.
class _RateLimitHit(BaseException):
    """Raised internally when Groq returns 429 — signals the cycle to abort."""
    pass


class _BudgetExhausted(BaseException):
    """Raised when the Dream Engine's own daily token budget is spent."""
    pass


class _BaseCycle:
    """Common init shared by all five cycles."""

    # Class-level circuit breaker shared across ALL cycles in a dream run.
    # Once Groq returns a 429 (daily token budget exhausted), there is no point
    # firing dozens more calls that will all fail — they just spam the logs and
    # waste time. The DreamEngine resets this to False at the start of each run.
    _rate_limited: bool = False

    # Shared daily-token-budget guard, injected by the DreamEngine at startup.
    # When set, every dream LLM call is gated on it so dreaming can never drain
    # the org's whole daily Groq allowance and starve the live voice agent.
    _budget: "DreamTokenBudget | None" = None

    def __init__(
        self,
        store:              QdrantStore,
        trace_store:        "ExecutionTraceStore",
        groq_client:        "AsyncGroq",
        retrieval_pipeline: "RetrievalPipeline | None" = None,
    ) -> None:
        self._store              = store
        self._trace_store        = trace_store
        self._groq               = groq_client
        self._retrieval_pipeline = retrieval_pipeline

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        """True if this exception is a Groq 429 rate-limit error."""
        return "429" in str(exc) or "rate_limit" in str(exc).lower()

    def _check_budget(self, max_tokens: int) -> None:
        """Raise _BudgetExhausted if this call would exceed the dream budget."""
        if _BaseCycle._budget is not None and not _BaseCycle._budget.can_afford(max_tokens):
            raise _BudgetExhausted()

    async def _record_usage(self, resp) -> None:
        """Record actual tokens consumed by a Groq response into the budget."""
        if _BaseCycle._budget is None:
            return
        try:
            used = getattr(resp, "usage", None)
            total = getattr(used, "total_tokens", 0) if used else 0
            await _BaseCycle._budget.record(int(total))
        except Exception as exc:
            logger.warning(f"_record_usage failed (non-fatal): {exc}")

    # Convenience: call LLM and parse JSON response
    async def _llm_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
    ) -> dict:
        """
        Send a prompt to the dream LLM and parse the JSON response.
        Returns an empty dict on failure — never raises.

        Aborts the whole cycle on a 429 (_RateLimitHit) or when the Dream
        Engine's own daily token budget is spent (_BudgetExhausted), so it
        never drains the org allowance the voice agent depends on.
        """
        if _BaseCycle._rate_limited:
            raise _RateLimitHit()
        self._check_budget(max_tokens)   # raises _BudgetExhausted if over cap
        try:
            resp = await self._groq.chat.completions.create(
                model=DREAM_LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                # GPT-OSS reasoning model: cap reasoning so max_tokens isn't
                # entirely consumed by hidden chain-of-thought (which would
                # return empty JSON and waste dream budget).
                reasoning_effort="low",
                max_tokens=max_tokens,
                temperature=0.3,
            )
            await self._record_usage(resp)
            raw = resp.choices[0].message.content or "{}"
            # Strip markdown code fences if LLM wraps JSON in ```
            raw = raw.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1]) if len(lines) > 2 else "{}"
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(f"_llm_json: JSON parse failed: {exc}")
            return {}
        except Exception as exc:
            if self._is_rate_limit(exc):
                _BaseCycle._rate_limited = True
                logger.warning("_llm_json: Groq 429 (daily token budget hit) — aborting dream cycle")
                raise _RateLimitHit() from exc
            logger.warning(f"_llm_json: LLM call failed (non-fatal): {exc}")
            return {}

    async def _llm_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 256,
    ) -> str:
        """Plain text LLM call. Returns empty string on failure."""
        if _BaseCycle._rate_limited:
            raise _RateLimitHit()
        self._check_budget(max_tokens)   # raises _BudgetExhausted if over cap
        try:
            resp = await self._groq.chat.completions.create(
                model=DREAM_LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                reasoning_effort="low",   # GPT-OSS reasoning model — minimise overhead
                max_tokens=max_tokens,
                temperature=0.4,
            )
            await self._record_usage(resp)
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            if self._is_rate_limit(exc):
                _BaseCycle._rate_limited = True
                logger.warning("_llm_text: Groq 429 (daily token budget hit) — aborting dream cycle")
                raise _RateLimitHit() from exc
            logger.warning(f"_llm_text: LLM call failed (non-fatal): {exc}")
            return ""


# ---------------------------------------------------------------------------
# Cycle 1: FailureAnalysisCycle
# ---------------------------------------------------------------------------

_EVALUATOR_SYSTEM = """You are a SALES-QUALITY evaluator for BharatConnect's AI voice sales agent.
This agent is a SALES agent, not a passive FAQ bot. Its job is to help the customer AND move them toward becoming a lead/customer. Judge each turn on BOTH: did it serve the customer well, AND did it advance the sale?

Evaluate the conversation turn below and return ONLY valid JSON (no markdown, no explanation).

Score each dimension 0-10, then give an overall score.

Dimensions:
  - correctness:   accurate, grounded in the knowledge base, no invented facts.
  - helpfulness:   directly answered the customer's actual need; empathetic, concise, natural to hear aloud.
  - sales_progress: did this turn ADVANCE the sale? Credit: answering then guiding (not parking with "anything else?"), qualifying the need, pitching a relevant plan with its price, handling an objection, or attempting a soft close ("shall I register your interest?") at the right moment. A turn that merely answered and stalled scores LOW here even if the answer was correct.

Overall score (1-10) reflects BOTH serving the customer and advancing the sale:
  10 = excellent: correct, helpful, AND moved the customer one clear step toward converting.
   8 = good: solid answer that also nudged toward the goal, minor issues.
   6 = adequate: correct/helpful but did NOT advance the sale (e.g. answered then asked "anything else?").
   4 = poor: vague, missed the need, OR ignored an obvious chance to qualify/pitch/close.
   2 = bad: wrong facts, hallucination, no relevant answer, or actively pushed the customer away.
   1 = critical: harmful, wrong policy, asked for forbidden info, or completely off-topic.
Do NOT penalize sales_progress when the turn is a pure greeting, a goodbye/closing turn, or when advancing would be unnatural (customer is mid-question, upset, or explicitly ending). Judge in context.

Return this exact structure:
{
  "score": <int 1-10>,
  "dimensions": {"correctness": <int 0-10>, "helpfulness": <int 0-10>, "sales_progress": <int 0-10>},
  "issues": ["<issue1>", "<issue2>"],
  "suggested_improvement": "<one concrete sentence, sales-aware>"
}"""


class FailureAnalysisCycle(_BaseCycle):
    """
    Cycle 1 — Score every unprocessed conversation turn 1-10.

    Reads turns in batches of 10 from execution_traces where
    dream_processed=False.  Calls the LLM evaluator on each.
    Writes eval_score + issues back via update_eval_score().
    Saves checkpoint after each batch.
    """

    CYCLE_TYPE = "failure_analysis"
    BATCH_SIZE = 10

    async def run(self, pause_signal: asyncio.Event) -> None:
        logger.info("FailureAnalysisCycle: starting")

        checkpoint = await _load_checkpoint(self._store, self.CYCLE_TYPE)
        processed_count = checkpoint.get("processed_count", 0) if checkpoint else 0
        logger.info(
            f"FailureAnalysisCycle: resuming from checkpoint — "
            f"already processed {processed_count} turns"
        )

        total_processed = 0

        while True:
            if pause_signal.is_set():
                await _save_checkpoint(
                    self._store, self.CYCLE_TYPE,
                    {"processed_count": processed_count + total_processed},
                )
                logger.info(
                    f"FailureAnalysisCycle: PAUSED after {total_processed} turns "
                    "(customer connected)"
                )
                return

            try:
                turns = await self._trace_store.get_unprocessed_turns(
                    limit=self.BATCH_SIZE
                )
            except Exception as exc:
                logger.error(f"FailureAnalysisCycle: fetch failed: {exc}")
                break

            if not turns:
                logger.info("FailureAnalysisCycle: no unprocessed turns — done")
                break

            for turn in turns:
                if pause_signal.is_set():
                    await _save_checkpoint(
                        self._store, self.CYCLE_TYPE,
                        {"processed_count": processed_count + total_processed},
                    )
                    logger.info("FailureAnalysisCycle: PAUSED mid-batch")
                    return

                await self._score_turn(turn)
                total_processed += 1

            logger.info(
                f"FailureAnalysisCycle: batch done — total this run: {total_processed}"
            )

        # Completed all unprocessed turns.
        await _save_checkpoint(
            self._store, self.CYCLE_TYPE,
            {"processed_count": processed_count + total_processed},
            status="done",
        )
        logger.info(
            f"FailureAnalysisCycle: DONE — scored {total_processed} turns this run"
        )

    async def _score_turn(self, turn: "TurnTrace") -> None:
        """Score one turn and write the result back to Qdrant."""
        rag_summary = (
            ", ".join(d.get("doc_id", "?")[:20] for d in turn.retrieved_docs)
            if turn.retrieved_docs else "none"
        )
        user_prompt = (
            f"User input: {turn.user_input}\n"
            f"Retrieved context doc IDs: {rag_summary}\n"
            f"AI response: {turn.ai_response}\n"
            f"Latency: {turn.latency_ms}ms\n"
            f"Emotion hint: {turn.emotion_hint}"
        )

        result = await self._llm_json(_EVALUATOR_SYSTEM, user_prompt)
        score  = float(result.get("score", 5))
        issues = result.get("issues", [])
        dimensions = result.get("dimensions") or None

        try:
            await self._trace_store.update_eval_score(
                session_id  = turn.session_id,
                turn_index  = turn.turn_index,
                score       = score,
                issues      = issues,
                dimensions  = dimensions,
            )
            logger.debug(
                f"FailureAnalysisCycle: scored session={turn.session_id[:8]}… "
                f"turn={turn.turn_index} score={score:.1f} issues={issues}"
            )
        except Exception as exc:
            logger.warning(f"FailureAnalysisCycle: update_eval_score failed: {exc}")


# ---------------------------------------------------------------------------
# Cycle 2: RetrievalQualityAnalysisCycle
# ---------------------------------------------------------------------------

_RETRIEVAL_EVAL_SYSTEM = """You are an expert at evaluating RAG (Retrieval-Augmented Generation) quality.
Analyze whether the retrieved context was useful for answering the user's question.
Return ONLY valid JSON (no markdown, no explanation):
{
  "retrieval_was_relevant": <true|false>,
  "relevance_score": <float 0.0-1.0>,
  "better_query": "<improved search query that would have found better context, or empty string>",
  "knowledge_gap": "<description of missing knowledge if no relevant doc existed, or empty string>",
  "suggested_doc": "<what document/FAQ entry would fill the gap, or empty string>"
}"""


class RetrievalQualityAnalysisCycle(_BaseCycle):
    """
    Cycle 2 — Assess RAG quality on poor-scoring turns (eval_score < 6).

    For each low-scoring turn:
      - Judges whether retrieved context was relevant.
      - If a knowledge gap is detected, logs it to improvement_log.
      - Stores query-reformulation hints for future tuning.
    """

    CYCLE_TYPE       = "retrieval_quality"
    SCORE_CUTOFF     = 6.0
    BATCH_SIZE       = 20
    PROCESSED_MARKER = "dream_retrieval_analysed"

    async def run(self, pause_signal: asyncio.Event) -> None:
        logger.info("RetrievalQualityAnalysisCycle: starting")

        # Scroll execution_traces for scored, poor turns.
        try:
            from qdrant_client.models import FieldCondition, Filter, Range

            low_score_filter = Filter(
                must=[
                    FieldCondition(key="dream_processed", match={"value": True}),
                ]
            )
            records, _ = await self._store.scroll(
                EXECUTION_TRACES,
                filter=low_score_filter,
                limit=self.BATCH_SIZE * 5,
            )
        except Exception as exc:
            logger.error(f"RetrievalQualityAnalysisCycle: scroll failed: {exc}")
            return

        # Filter in Python — Qdrant free tier doesn't support float range index.
        # Skip turns already analysed by THIS cycle (marker in payload) so each
        # idle window doesn't re-issue identical LLM calls on the same turns
        # (Bug #8 — repeated shared-budget token spend).
        poor_turns = [
            r for r in records
            if isinstance(r["payload"].get("eval_score"), (int, float))
            and r["payload"]["eval_score"] < self.SCORE_CUTOFF
            and not r["payload"].get(self.PROCESSED_MARKER)
        ]
        logger.info(
            f"RetrievalQualityAnalysisCycle: {len(poor_turns)} poor turns to analyse"
        )

        gap_count = 0
        for i, record in enumerate(poor_turns):
            if pause_signal.is_set():
                await _save_checkpoint(
                    self._store, self.CYCLE_TYPE,
                    {"analysed_index": i},
                )
                logger.info("RetrievalQualityAnalysisCycle: PAUSED mid-run")
                return

            await self._analyse_turn(record["payload"])
            # Mark analysed so a later rotation skips it (non-fatal on failure).
            try:
                await self._store.update_payload(
                    EXECUTION_TRACES, str(record["id"]),
                    {self.PROCESSED_MARKER: True},
                )
            except Exception as exc:
                logger.warning(f"RetrievalQualityAnalysisCycle: mark failed (non-fatal): {exc}")
            gap_count += 1

        await _save_checkpoint(
            self._store, self.CYCLE_TYPE,
            {"analysed_index": len(poor_turns)},
            status="done",
        )
        logger.info(
            f"RetrievalQualityAnalysisCycle: DONE — analysed {gap_count} poor turns"
        )

    async def _analyse_turn(self, payload: dict) -> None:
        user_input     = payload.get("user_input", "")
        ai_response    = payload.get("ai_response", "")
        retrieved_docs = payload.get("retrieved_docs", [])
        eval_score     = payload.get("eval_score", 0)

        rag_snippet = "\n".join(
            f"- [{d.get('doc_id','?')[:30]}] {d.get('content','')[:200]}"
            for d in retrieved_docs
        ) or "(no documents retrieved)"

        user_prompt = (
            f"User asked: {user_input}\n"
            f"AI answered: {ai_response}\n"
            f"Turn quality score: {eval_score}/10\n"
            f"Retrieved context:\n{rag_snippet}"
        )

        result = await self._llm_json(_RETRIEVAL_EVAL_SYSTEM, user_prompt)

        # Log a knowledge gap if detected.
        gap = result.get("knowledge_gap", "").strip()
        if gap:
            desc = f"KNOWLEDGE GAP: {gap} | Suggested doc: {result.get('suggested_doc', '')}"
            await _log_improvement(
                self._store,
                category    = "knowledge_gap",
                description = desc,
                extra       = {
                    "user_input":     user_input,
                    "suggested_doc":  result.get("suggested_doc", ""),
                    "better_query":   result.get("better_query", ""),
                },
            )
            logger.info(f"RetrievalQualityAnalysisCycle: knowledge gap logged — {gap[:80]}")

        # Log query reformulation hint.
        better_query = result.get("better_query", "").strip()
        if better_query and not result.get("retrieval_was_relevant", True):
            await _log_improvement(
                self._store,
                category    = "retrieval",
                description = (
                    f"Query reformulation: '{user_input}' → '{better_query}'"
                ),
                extra = {"original_query": user_input, "improved_query": better_query},
            )


# ---------------------------------------------------------------------------
# Cycle 3: PromptImprovementCycle
# ---------------------------------------------------------------------------

_PROMPT_PROPOSAL_SYSTEM = """You are an expert at improving the system prompt of a SALES voice agent for BharatConnect (a telecom). The agent must help customers AND convert them into leads/customers.
You will be shown conversation turns where the agent scored poorly (under 6/10) — often because it answered correctly but FAILED TO ADVANCE THE SALE (didn't qualify, pitch, or move toward a close), or was unhelpful.
Propose ONE instruction that, if added to the system prompt, would improve BOTH customer experience AND sales conversion on these cases. Prefer instructions that make the agent guide the customer one concrete step toward becoming a lead, without being pushy or inventing facts.
Return ONLY valid JSON (no markdown):
{
  "addendum": "<the exact instruction text to add to the system prompt>",
  "applies_to": "<brief description of what failure pattern this fixes>",
  "topic": "billing|network|policy|competitive|general",
  "confidence": <float 0.0-1.0>
}"""

_PROMPT_JUDGE_SYSTEM = """You are a judge for a SALES voice agent. Assess whether a proposed system-prompt instruction would have led to a BETTER turn — better meaning both more helpful to the customer AND more likely to advance them toward converting (qualify / pitch relevantly / soft close), without pushiness or invented facts.
Given the failing turn and the proposed instruction, judge honestly — reject instructions that would help answer quality but hurt sales flow, or vice versa, or that contradict good conversational tone.
Return ONLY valid JSON (no markdown):
{
  "would_improve": <true|false>,
  "improvement_score": <float 0.0-1.0>,
  "reason": "<one sentence>"
}"""


class PromptImprovementCycle(_BaseCycle):
    """
    Cycle 3 — Propose system prompt improvements from failure clusters.

    Groups poor-scoring turns into clusters of 5, asks the LLM to propose one
    improvement per cluster, then judges the proposal against held-out turns.
    Approved improvements (judge score ≥ 0.6) are stored in improvement_log
    with category="prompt".

    stream_agent() reads approved addenda from improvement_log at call time and
    appends them to BASE_SYSTEM_PROMPT — so every subsequent call benefits.
    """

    CYCLE_TYPE        = "prompt_improvement"
    SCORE_CUTOFF      = 6.0
    CLUSTER_SIZE      = 5
    MIN_JUDGE_SCORE   = 0.6
    PROCESSED_MARKER  = "dream_prompt_clustered"

    async def _mark_processed(self, records: list[dict]) -> None:
        """Mark each trace processed by this cycle so it isn't re-clustered."""
        for r in records:
            try:
                await self._store.update_payload(
                    EXECUTION_TRACES, str(r["id"]),
                    {self.PROCESSED_MARKER: True},
                )
            except Exception as exc:
                logger.warning(f"PromptImprovementCycle: mark failed (non-fatal): {exc}")

    async def run(self, pause_signal: asyncio.Event) -> None:
        logger.info("PromptImprovementCycle: starting")

        # Gather poor turns.
        try:
            records, _ = await self._store.scroll(
                EXECUTION_TRACES,
                filter=self._store.filter_eq("dream_processed", True),
                limit=100,
            )
        except Exception as exc:
            logger.error(f"PromptImprovementCycle: scroll failed: {exc}")
            return

        # Keep full records (need ids to mark processed). Skip turns already
        # clustered by this cycle so a later rotation doesn't re-issue the same
        # LLM proposals/judges on them (Bug #8).
        poor = [
            r for r in records
            if isinstance(r["payload"].get("eval_score"), (int, float))
            and r["payload"]["eval_score"] < self.SCORE_CUTOFF
            and not r["payload"].get(self.PROCESSED_MARKER)
        ]

        if not poor:
            logger.info("PromptImprovementCycle: no poor turns — nothing to do")
            await _save_checkpoint(
                self._store, self.CYCLE_TYPE, {}, status="done"
            )
            return

        logger.info(
            f"PromptImprovementCycle: {len(poor)} poor turns → "
            f"{len(poor) // self.CLUSTER_SIZE} clusters"
        )

        # Simple grouping into fixed-size clusters (no external KMeans needed).
        clusters = [
            poor[i:i + self.CLUSTER_SIZE]
            for i in range(0, len(poor), self.CLUSTER_SIZE)
        ]

        improvements_applied = 0

        for cluster_idx, cluster_recs in enumerate(clusters):
            if pause_signal.is_set():
                await _save_checkpoint(
                    self._store, self.CYCLE_TYPE,
                    {"cluster_idx": cluster_idx},
                )
                logger.info("PromptImprovementCycle: PAUSED")
                return

            cluster = [r["payload"] for r in cluster_recs]
            # Mark every turn in this cluster processed — it's about to be
            # analysed, so it must not be re-clustered next rotation.
            await self._mark_processed(cluster_recs)

            proposal = await self._propose(cluster)
            if not proposal.get("addendum"):
                continue

            confidence = float(proposal.get("confidence", 0.0))
            if confidence < 0.4:
                logger.debug(
                    f"PromptImprovementCycle: low-confidence proposal skipped "
                    f"(confidence={confidence:.2f})"
                )
                continue

            # Judge the proposal.
            judge_scores = []
            for held_out in cluster[:3]:
                verdict = await self._judge(held_out, proposal["addendum"])
                judge_scores.append(float(verdict.get("improvement_score", 0)))

            avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0
            logger.info(
                f"PromptImprovementCycle: cluster {cluster_idx} — "
                f"judge_avg={avg_judge:.2f} confidence={confidence:.2f}"
            )

            if avg_judge >= self.MIN_JUDGE_SCORE:
                await _log_improvement(
                    self._store,
                    category      = "prompt",
                    description   = proposal["addendum"],
                    before_metric = 0.0,
                    after_metric  = avg_judge,
                    extra         = {
                        "applies_to":  proposal.get("applies_to", ""),
                        "confidence":  confidence,
                        "judge_score": avg_judge,
                        "approved":    True,
                        "topic":       proposal.get("topic", "general"),
                    },
                )
                improvements_applied += 1
                logger.info(
                    f"PromptImprovementCycle: improvement APPROVED and logged — "
                    f"{proposal['addendum'][:80]!r}"
                )

        await _save_checkpoint(
            self._store, self.CYCLE_TYPE,
            {"improvements_applied": improvements_applied},
            status="done",
        )
        logger.info(
            f"PromptImprovementCycle: DONE — {improvements_applied} improvements logged"
        )

    async def _propose(self, cluster: list[dict]) -> dict:
        turns_text = "\n\n".join(
            f"Turn {i+1} (score {t.get('eval_score', '?')}/10):\n"
            f"  User: {t.get('user_input', '')}\n"
            f"  AI:   {t.get('ai_response', '')}"
            for i, t in enumerate(cluster)
        )
        return await self._llm_json(
            _PROMPT_PROPOSAL_SYSTEM,
            f"Poor-performing turns:\n\n{turns_text}",
            max_tokens=256,
        )

    async def _judge(self, turn: dict, addendum: str) -> dict:
        user_prompt = (
            f"Failing turn:\n"
            f"  User: {turn.get('user_input', '')}\n"
            f"  AI:   {turn.get('ai_response', '')}\n"
            f"  Score: {turn.get('eval_score', '?')}/10\n\n"
            f"Proposed instruction to add:\n\"{addendum}\""
        )
        return await self._llm_json(_PROMPT_JUDGE_SYSTEM, user_prompt)


# ---------------------------------------------------------------------------
# Cycle 4: SyntheticQueryGenCycle
# ---------------------------------------------------------------------------

_SYNTHETIC_SYSTEM = """You are a data generation expert for AI training.
Generate realistic customer support conversation queries based on the pattern shown.
Return ONLY valid JSON (no markdown):
{
  "queries": [
    {"user_input": "<query1>", "type": "adversarial|edge_case|happy_path"},
    {"user_input": "<query2>", "type": "adversarial|edge_case|happy_path"},
    {"user_input": "<query3>", "type": "adversarial|edge_case|happy_path"}
  ]
}"""


class SyntheticQueryGenCycle(_BaseCycle):
    """
    Cycle 4 — Generate synthetic test conversations from real failure patterns.

    Creates 3 variations per poor turn:
      - adversarial: same question phrased differently / more aggressively
      - edge_case:   related question the bot might also struggle with
      - happy_path:  successful variant to keep the distribution balanced

    Synthetic traces are stored in execution_traces with
    customer_id="synthetic" so they're excluded from real metrics but
    available to FailureAnalysisCycle as regression-test material.
    """

    CYCLE_TYPE       = "synthetic_query_gen"
    SCORE_CUTOFF     = 6.0
    MAX_TURNS        = 30   # cap per run to avoid runaway generation
    PROCESSED_MARKER = "dream_synthetic_generated"

    async def run(self, pause_signal: asyncio.Event) -> None:
        logger.info("SyntheticQueryGenCycle: starting")

        try:
            records, _ = await self._store.scroll(
                EXECUTION_TRACES,
                filter=self._store.filter_eq("dream_processed", True),
                limit=self.MAX_TURNS * 3,
            )
        except Exception as exc:
            logger.error(f"SyntheticQueryGenCycle: scroll failed: {exc}")
            return

        # Keep full records (ids) and skip turns already used to generate
        # synthetics, so we don't regenerate variants for them every rotation
        # (Bug #8).
        poor = [
            r for r in records
            if isinstance(r["payload"].get("eval_score"), (int, float))
            and r["payload"]["eval_score"] < self.SCORE_CUTOFF
            and not r["payload"].get(self.PROCESSED_MARKER)
        ][:self.MAX_TURNS]

        if not poor:
            logger.info("SyntheticQueryGenCycle: no poor turns — skipping")
            await _save_checkpoint(
                self._store, self.CYCLE_TYPE, {}, status="done"
            )
            return

        generated = 0

        for i, record in enumerate(poor):
            if pause_signal.is_set():
                await _save_checkpoint(
                    self._store, self.CYCLE_TYPE,
                    {"generated": generated, "turn_index": i},
                )
                logger.info("SyntheticQueryGenCycle: PAUSED")
                return

            turn = record["payload"]
            queries = await self._generate_variants(turn)
            for q in queries:
                await self._store_synthetic_turn(
                    user_input = q.get("user_input", ""),
                    query_type = q.get("type", "synthetic"),
                    source_turn = turn,
                )
                generated += 1
            # Mark the source turn so its variants aren't regenerated next run.
            try:
                await self._store.update_payload(
                    EXECUTION_TRACES, str(record["id"]),
                    {self.PROCESSED_MARKER: True},
                )
            except Exception as exc:
                logger.warning(f"SyntheticQueryGenCycle: mark failed (non-fatal): {exc}")

        await _save_checkpoint(
            self._store, self.CYCLE_TYPE,
            {"generated": generated},
            status="done",
        )
        logger.info(f"SyntheticQueryGenCycle: DONE — generated {generated} synthetic queries")

    async def _generate_variants(self, turn: dict) -> list[dict]:
        user_prompt = (
            f"Original failing query (score {turn.get('eval_score', '?')}/10):\n"
            f"User: {turn.get('user_input', '')}\n"
            f"AI: {turn.get('ai_response', '')}\n\n"
            "Generate 3 related test queries (one adversarial, one edge_case, one happy_path)."
        )
        result = await self._llm_json(_SYNTHETIC_SYSTEM, user_prompt)
        return result.get("queries", [])

    async def _store_synthetic_turn(
        self,
        user_input:  str,
        query_type:  str,
        source_turn: dict,
    ) -> None:
        if not user_input.strip():
            return

        synthetic_session = f"synthetic_{uuid.uuid4().hex[:12]}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{synthetic_session}:0"))

        payload = {
            "session_id":        synthetic_session,
            "turn_index":        0,
            "user_input":        user_input,
            "detected_language": source_turn.get("detected_language", "en-IN"),
            "retrieved_docs":    [],
            "tool_calls":        [],
            "ai_response":       "",
            "latency_ms":        0,
            "emotion_hint":      "neutral",
            "created_at":        _utc_now(),
            "eval_score":        None,
            "customer_feedback": None,
            "dream_processed":   False,
            "customer_id":       "synthetic",
            "synthetic_type":    query_type,
            "source_session":    source_turn.get("session_id", ""),
        }

        try:
            await self._store.upsert(
                EXECUTION_TRACES,
                [{"id": point_id, "vector": self._store.dummy_vector(), "payload": payload}],
            )
        except Exception as exc:
            logger.warning(f"SyntheticQueryGenCycle: store synthetic turn failed: {exc}")


# ---------------------------------------------------------------------------
# Cycle 5: MemoryConsolidationCycle
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM = """You are a customer support analyst.
Summarize this customer's support history in 2-3 concise sentences.
Focus on: main issues raised, resolutions provided, any ongoing concerns, preferred language.
Return plain text only — no JSON, no bullet points."""


class MemoryConsolidationCycle(_BaseCycle):
    """
    Cycle 5 — Housekeeping and long-term memory management.

    1. Stale profile cleanup: customers not seen in 90 days → mark stale.
    2. Summary refresh: customers with ≥ 5 sessions → regenerate issue summary.
    3. KB hygiene: knowledge chunks never retrieved → flag in improvement_log.
    4. Synthetic turn cleanup: remove synthetic traces older than 30 days.
    """

    CYCLE_TYPE               = "memory_consolidation"
    STALE_DAYS               = 90
    SUMMARY_REFRESH_SESSIONS = 5
    SYNTHETIC_CLEANUP_DAYS   = 30

    async def run(self, pause_signal: asyncio.Event) -> None:
        logger.info("MemoryConsolidationCycle: starting")

        steps = [
            ("stale_profiles",   self._cleanup_stale_profiles),
            ("summary_refresh",  self._refresh_customer_summaries),
            ("kb_hygiene",       self._flag_unused_kb_docs),
            ("cleanup_traces",   self._cleanup_old_traces),
        ]

        for step_name, step_fn in steps:
            if pause_signal.is_set():
                await _save_checkpoint(
                    self._store, self.CYCLE_TYPE,
                    {"paused_at": step_name},
                )
                logger.info(f"MemoryConsolidationCycle: PAUSED before '{step_name}'")
                return

            try:
                await step_fn()
            except Exception as exc:
                logger.error(
                    f"MemoryConsolidationCycle: step '{step_name}' failed (skipping): {exc}"
                )

        await _save_checkpoint(
            self._store, self.CYCLE_TYPE, {}, status="done"
        )
        logger.info("MemoryConsolidationCycle: DONE")

    async def _cleanup_stale_profiles(self) -> None:
        """Mark customer profiles as stale if not seen in STALE_DAYS days."""
        from datetime import timedelta

        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=self.STALE_DAYS)
        ).isoformat()

        records, _ = await self._store.scroll(
            CUSTOMER_PROFILES, limit=500
        )
        stale_count = 0

        for r in records:
            last_seen = r["payload"].get("last_seen_at", "")
            if last_seen and last_seen < cutoff:
                try:
                    await self._store.update_payload(
                        CUSTOMER_PROFILES,
                        str(r["id"]),
                        {"stale": True},
                    )
                    stale_count += 1
                except Exception as exc:
                    logger.warning(f"MemoryConsolidationCycle: stale mark failed: {exc}")

        logger.info(
            f"MemoryConsolidationCycle: marked {stale_count} profiles as stale "
            f"(not seen in {self.STALE_DAYS} days)"
        )

    async def _refresh_customer_summaries(self) -> None:
        """
        Regenerate past_issue_summary for customers with many sessions.
        Only processes customers who have ≥ SUMMARY_REFRESH_SESSIONS distinct
        session IDs in execution_traces.
        """
        records, _ = await self._store.scroll(
            CUSTOMER_PROFILES, limit=200
        )
        refreshed = 0

        for r in records:
            customer_id = r["payload"].get("customer_id", "")
            if not customer_id or customer_id == "anonymous":
                continue

            # Count sessions for this customer by scanning traces.
            try:
                trace_records, _ = await self._store.scroll(
                    EXECUTION_TRACES,
                    filter=self._store.filter_eq("session_id", customer_id),
                    limit=10,
                )
            except Exception:
                continue

            if len(trace_records) < self.SUMMARY_REFRESH_SESSIONS:
                continue

            # Build a short history text from recent turns.
            recent = sorted(
                trace_records,
                key=lambda x: x["payload"].get("created_at", ""),
                reverse=True,
            )[:10]

            history_text = "\n".join(
                f"User: {t['payload'].get('user_input','')}\n"
                f"AI: {t['payload'].get('ai_response','')}"
                for t in recent
            )

            new_summary = await self._llm_text(
                _SUMMARY_SYSTEM,
                f"Customer ID: {customer_id[:16]}\n\nRecent conversations:\n{history_text}",
                max_tokens=150,
            )

            if new_summary:
                try:
                    await self._store.update_payload(
                        CUSTOMER_PROFILES,
                        str(r["id"]),
                        {"past_issue_summary": new_summary},
                    )
                    refreshed += 1
                except Exception as exc:
                    logger.warning(
                        f"MemoryConsolidationCycle: summary refresh failed for "
                        f"{customer_id[:16]}: {exc}"
                    )

        logger.info(
            f"MemoryConsolidationCycle: refreshed summaries for {refreshed} customers"
        )

    async def _flag_unused_kb_docs(self) -> None:
        """
        Flag knowledge base documents that were never retrieved.
        These appear in improvement_log as category="knowledge_gap" so the
        operator knows which docs might be poorly indexed or irrelevant.
        """
        try:
            kb_records, _ = await self._store.scroll(
                KNOWLEDGE_BASE, limit=500
            )
        except Exception as exc:
            logger.warning(f"MemoryConsolidationCycle: KB scan failed: {exc}")
            return

        if not kb_records:
            return

        # Collect doc_ids that appear in any execution trace.
        try:
            trace_records, _ = await self._store.scroll(
                EXECUTION_TRACES, limit=1000
            )
        except Exception:
            return

        retrieved_ids: set[str] = set()
        for tr in trace_records:
            for doc in tr["payload"].get("retrieved_docs", []):
                if isinstance(doc, dict):
                    retrieved_ids.add(doc.get("doc_id", ""))

        flagged = 0
        for kb in kb_records:
            doc_id = kb["payload"].get("doc_id", "")
            if doc_id and doc_id not in retrieved_ids:
                await _log_improvement(
                    self._store,
                    category    = "knowledge_gap",
                    description = (
                        f"KB document never retrieved in any conversation: "
                        f"doc_id={doc_id} source={kb['payload'].get('source','?')}"
                    ),
                    extra = {
                        "doc_id": doc_id,
                        "source": kb["payload"].get("source", ""),
                        "reason": "never_retrieved",
                    },
                )
                flagged += 1

        logger.info(
            f"MemoryConsolidationCycle: flagged {flagged} never-retrieved KB documents"
        )

    async def _cleanup_old_traces(self) -> None:
        """
        Delete synthetic traces and fully processed real traces that are older than 30 days
        to free up space in our free-tier Qdrant database.
        """
        from datetime import timedelta

        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=self.SYNTHETIC_CLEANUP_DAYS)
        ).isoformat()

        # Fetch traces from EXECUTION_TRACES (limit 500 per run to avoid memory spikes)
        try:
            records, _ = await self._store.scroll(
                EXECUTION_TRACES,
                limit=500
            )
        except Exception as exc:
            logger.warning(f"MemoryConsolidationCycle: scroll for cleanup failed: {exc}")
            return

        ids_to_delete = []
        for r in records:
            payload = r["payload"]
            created_at = payload.get("created_at", "")
            if not created_at or created_at >= cutoff:
                continue

            # Case 1: Synthetic trace older than 30 days
            is_synthetic = payload.get("customer_id") == "synthetic"

            # Case 2: Real trace fully processed and older than 30 days
            is_processed_real = payload.get("dream_processed") is True

            if is_synthetic or is_processed_real:
                ids_to_delete.append(str(r["id"]))

        if ids_to_delete:
            try:
                await self._store.delete_points(EXECUTION_TRACES, ids_to_delete)
                logger.info(
                    f"MemoryConsolidationCycle: deleted {len(ids_to_delete)} old/synthetic traces "
                    f"older than {self.SYNTHETIC_CLEANUP_DAYS} days"
                )
            except Exception as exc:
                logger.warning(f"MemoryConsolidationCycle: failed to delete old traces: {exc}")
        else:
            logger.info("MemoryConsolidationCycle: no old/synthetic traces need deletion")


# ---------------------------------------------------------------------------
# Exported names for engine.py
# ---------------------------------------------------------------------------

__all__ = [
    "FailureAnalysisCycle",
    "RetrievalQualityAnalysisCycle",
    "PromptImprovementCycle",
    "SyntheticQueryGenCycle",
    "MemoryConsolidationCycle",
]
