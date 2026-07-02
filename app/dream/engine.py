"""
dream/engine.py — Dream Engine  (Phase 5)
==========================================

The Dream Engine runs as a permanent background asyncio.Task for the lifetime
of the FastAPI process.  Its job: when no customer is on a call, improve the
bot by running five analysis sub-cycles back-to-back.

State machine
─────────────
  PAUSED   ←──────────────────── customer_connected()
     │
     │  (active_sessions drops to 0 + DREAM_IDLE_THRESHOLD_SECS elapses)
     ▼
  DREAMING  ──→  cycle 1 → 2 → 3 → 4 → 5 → (repeat)
     │
     │  customer_connected() fires at any moment
     ▼
  PAUSED   (current cycle saves checkpoint, returns immediately)

The pause mechanism uses an asyncio.Event:
  _pause_event.set()   = dream is PAUSED  (event is "signalled")
  _pause_event.clear() = dream is RUNNING (event is "clear")

Each cycle's inner loop calls `pause_event.is_set()` after every work unit.
If it is set, the cycle saves a Qdrant checkpoint and returns — picking up from
that checkpoint on the next idle window.

Concurrency
───────────
  customer_connected() and customer_disconnected() are called from WebSocket
  coroutines (same event loop thread) — no lock needed; _active_sessions is an
  ordinary int and Python's GIL makes += / -= atomic at the bytecode level.

  The _resume_handle (asyncio.TimerHandle) is cancelled and recreated as
  needed; it is always replaced atomically inside the single event loop.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from app.config import DREAM_CYCLE_INTERVAL_SECS, DREAM_IDLE_THRESHOLD_SECS

if TYPE_CHECKING:
    from app.store import QdrantStore
    from app.knowledge.retriever import RetrievalPipeline
    from app.tracing.trace_store import ExecutionTraceStore
    from groq import AsyncGroq


class DreamEngine:
    """
    Background Dream Engine.

    Usage (from main.py lifespan):
        engine = DreamEngine(store, retrieval_pipeline, trace_store, groq_client)
        await engine.start()                 # starts background task, begins paused
        ...
        engine.customer_connected()          # call on every WS connect
        engine.customer_disconnected()       # call on every WS disconnect / finally
        ...
        await engine.stop()                  # graceful shutdown on app teardown
    """

    def __init__(
        self,
        store:              "QdrantStore",
        retrieval_pipeline: "RetrievalPipeline",
        trace_store:        "ExecutionTraceStore",
        groq_client:        "AsyncGroq",
    ) -> None:
        self._store              = store
        self._retrieval_pipeline = retrieval_pipeline
        self._trace_store        = trace_store
        self._groq               = groq_client

        # Hard daily token budget guard — caps how much the Dream Engine can
        # spend per UTC day so it can never drain the org allowance the live
        # voice agent depends on. Loaded from Qdrant in start().
        from app.dream.budget import DreamTokenBudget
        self._budget = DreamTokenBudget(store)

        self._active_sessions: int              = 0
        self._pause_event:     asyncio.Event    = asyncio.Event()   # set = paused
        self._dream_task:      asyncio.Task | None = None
        self._resume_handle:   asyncio.TimerHandle | None = None

        # Start paused — no customers are active at boot.
        self._pause_event.set()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API — called from main.py WebSocket handler
    # ──────────────────────────────────────────────────────────────────────────

    def customer_connected(self) -> None:
        """
        Call this immediately when a WebSocket connection is accepted.

        Increments the active customer counter and pauses the dream loop
        instantly.  If a resume timer was pending (from a prior disconnect),
        it is cancelled.
        """
        self._active_sessions += 1

        # Cancel any pending idle-delay timer.
        if self._resume_handle is not None:
            self._resume_handle.cancel()
            self._resume_handle = None

        # Signal the dream loop to pause.
        self._pause_event.set()
        logger.info(
            f"DreamEngine: customer connected — active_sessions={self._active_sessions}, dream PAUSED"
        )

    def customer_disconnected(self) -> None:
        """
        Call this in the WebSocket finally block when a connection ends.

        Decrements the active customer counter.  If it drops to zero, schedules
        a resume after DREAM_IDLE_THRESHOLD_SECS.  The delay avoids waking the
        dream engine for brief reconnects (page reload, mobile network blip).
        """
        self._active_sessions = max(0, self._active_sessions - 1)
        logger.info(
            f"DreamEngine: customer disconnected — active_sessions={self._active_sessions}"
        )

        if self._active_sessions == 0:
            self._schedule_resume()

    def status(self) -> dict:
        """Return engine status for the /health endpoint."""
        return {
            "active_sessions": self._active_sessions,
            "is_dreaming":     not self._pause_event.is_set(),
            "task_alive":      self._dream_task is not None and not self._dream_task.done(),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Start the background dream loop.

        Called once from the FastAPI lifespan after all sub-systems are ready.
        The loop starts paused; it wakes only when the first customer disconnects
        and DREAM_IDLE_THRESHOLD_SECS elapses.
        """
        if self._dream_task is not None and not self._dream_task.done():
            logger.warning("DreamEngine.start() called but task already running")
            return

        # Restore today's token spend so a restart mid-day can't reset the cap.
        await self._budget.load()
        # Share the budget guard with all cycles.
        from app.dream.cycles import _BaseCycle
        _BaseCycle._budget = self._budget

        self._pause_event.set()   # begin paused
        self._dream_task = asyncio.create_task(
            self._dream_loop(), name="dream-engine"
        )
        logger.info(
            f"DreamEngine started ✓  (initially paused — no customers yet) "
            f"| token budget: {self._budget.status()}"
        )

    async def stop(self) -> None:
        """
        Cancel the dream loop.  Called from FastAPI lifespan teardown.
        The loop is cancelled cleanly; any in-progress cycle will be interrupted.
        """
        if self._resume_handle is not None:
            self._resume_handle.cancel()

        if self._dream_task is not None and not self._dream_task.done():
            self._dream_task.cancel()
            try:
                await self._dream_task
            except asyncio.CancelledError:
                pass
        logger.info("DreamEngine stopped ✓")

    # ──────────────────────────────────────────────────────────────────────────
    # Internal — pause / resume
    # ──────────────────────────────────────────────────────────────────────────

    def _schedule_resume(self) -> None:
        """
        Schedule the dream loop to resume after the idle threshold.

        Uses call_later so the callback fires in the running event loop without
        an extra task.  The lambda guards against a reconnect that happened
        while the timer was pending — if _active_sessions rose back above 0,
        we do NOT clear the event (i.e. do NOT resume).
        """
        loop = asyncio.get_event_loop()
        self._resume_handle = loop.call_later(
            DREAM_IDLE_THRESHOLD_SECS,
            self._maybe_resume,
        )
        logger.info(
            f"DreamEngine: will resume in {DREAM_IDLE_THRESHOLD_SECS}s "
            "if no new customers connect"
        )

    def _maybe_resume(self) -> None:
        """
        Timer callback: resume the dream loop only if still idle.
        Runs in the event loop thread — safe to call asyncio methods here.
        """
        self._resume_handle = None
        if self._active_sessions == 0:
            self._pause_event.clear()   # un-pause
            logger.info("DreamEngine: idle threshold reached — dream loop RESUMED")
        else:
            logger.debug("DreamEngine: resume timer fired but customers still active — staying paused")

    # ──────────────────────────────────────────────────────────────────────────
    # Internal — dream loop
    # ──────────────────────────────────────────────────────────────────────────

    async def _dream_loop(self) -> None:
        """
        The main dream loop.  Runs forever until cancelled.

        Pattern:
          1. Wait until pause_event is cleared (i.e. resume signal).
          2. Run the next cycle in the rotation.
          3. Sleep DREAM_CYCLE_INTERVAL_SECS between cycles.
          4. Repeat.

        The pause_event check happens:
          - Here, between cycles (via _wait_for_idle()).
          - Inside each cycle, after every work unit (passed as pause_signal).
        """
        from app.dream.cycles import (
            FailureAnalysisCycle,
            MemoryConsolidationCycle,
            PromptImprovementCycle,
            RetrievalQualityAnalysisCycle,
            SyntheticQueryGenCycle,
            _BaseCycle,
            _BudgetExhausted,
            _RateLimitHit,
        )

        # When the daily Groq token budget is exhausted (either Groq's org-wide
        # 429, or our own self-imposed dream budget), back off this long before
        # retrying instead of the usual interval — the limit resets daily, so
        # retrying every few minutes just wastes cycles. 1h re-check picks the
        # budget back up if the UTC day rolled over or the tier was upgraded.
        RATE_LIMIT_BACKOFF_SECS = 3600  # 1 hour

        cycle_classes = [
            FailureAnalysisCycle,
            RetrievalQualityAnalysisCycle,
            PromptImprovementCycle,
            SyntheticQueryGenCycle,
            MemoryConsolidationCycle,
        ]
        cycle_index = 0

        logger.info("DreamEngine: _dream_loop entered")

        while True:
            try:
                # Wait here while paused (customers active or not yet idle enough).
                await self._wait_for_idle()

                CycleClass = cycle_classes[cycle_index % len(cycle_classes)]
                cycle_name = CycleClass.__name__

                logger.info(
                    f"DreamEngine: starting cycle [{cycle_index % len(cycle_classes) + 1}/"
                    f"{len(cycle_classes)}] {cycle_name}"
                )

                cycle = CycleClass(
                    store        = self._store,
                    trace_store  = self._trace_store,
                    groq_client  = self._groq,
                    retrieval_pipeline = self._retrieval_pipeline,
                )

                # Reset the shared rate-limit circuit breaker before each cycle
                # so a fresh run gets a clean attempt (the daily quota may have
                # reset since the last 429).
                _BaseCycle._rate_limited = False
                hit_rate_limit = False

                try:
                    await cycle.run(pause_signal=self._pause_event)
                except asyncio.CancelledError:
                    raise   # propagate cancel upward
                except _BudgetExhausted:
                    hit_rate_limit = True
                    logger.info(
                        f"DreamEngine: {cycle_name} stopped — dream daily token budget "
                        f"spent ({self._budget.status()}). Voice agent budget protected. "
                        f"Re-checking in {RATE_LIMIT_BACKOFF_SECS}s."
                    )
                except _RateLimitHit:
                    hit_rate_limit = True
                    logger.warning(
                        f"DreamEngine: {cycle_name} aborted — Groq daily token budget "
                        f"exhausted. Backing off {RATE_LIMIT_BACKOFF_SECS}s before retrying."
                    )
                except Exception as exc:
                    # Cycle failed — log and continue to next cycle rather than
                    # crashing the whole dream loop.
                    logger.error(f"DreamEngine: {cycle_name} raised an error (skipping): {exc}")

                cycle_index += 1

                # If we hit the daily token cap, back off for a long time —
                # retrying every 60s would just spam 429s until the quota resets.
                if hit_rate_limit:
                    await self._interruptible_sleep(RATE_LIMIT_BACKOFF_SECS)
                    continue

                # Respect the inter-cycle interval, but wake up immediately if
                # a customer connects (pause_event gets set).
                if not self._pause_event.is_set():
                    logger.debug(
                        f"DreamEngine: sleeping {DREAM_CYCLE_INTERVAL_SECS}s before next cycle"
                    )
                    await self._interruptible_sleep(DREAM_CYCLE_INTERVAL_SECS)

            except asyncio.CancelledError:
                logger.info("DreamEngine: _dream_loop cancelled (app shutdown)")
                return

    async def _wait_for_idle(self) -> None:
        """Block until the pause event is cleared (= system is idle)."""
        while self._pause_event.is_set():
            await asyncio.sleep(1)

    async def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep for `seconds` but wake up early if paused."""
        elapsed = 0.0
        step    = 2.0
        while elapsed < seconds:
            if self._pause_event.is_set():
                return   # customer connected — skip rest of sleep
            await asyncio.sleep(min(step, seconds - elapsed))
            elapsed += step
