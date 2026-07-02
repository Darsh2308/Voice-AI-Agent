"""
budget.py — Dream Engine daily token budget guard
===================================================

The Dream Engine and the live voice agent share ONE Groq organisation token
budget (free tier: 100K tokens/day). Left unchecked, a single night of dream
cycles can drain the entire daily allowance and 429 the voice agent — the
thing customers actually talk to.

This module enforces a HARD per-day token cap on dreaming so it can never
starve the voice path. It:

  • Tracks tokens spent by dream LLM calls within the current UTC day.
  • Refuses further dream calls once DREAM_DAILY_TOKEN_BUDGET is reached.
  • Resets automatically at the UTC date rollover.
  • Persists the running total to Qdrant (dream_checkpoints) so the cap
    survives a server restart — restarting does NOT reset the budget.

Accounting is based on Groq's reported usage (prompt + completion tokens)
from each response, so it reflects real consumption, not estimates.

Design note: this is a soft *pre-flight* reservation. Before each call we
check remaining budget against a conservative estimate (max_tokens for the
completion + a fixed prompt allowance). After the call we record the ACTUAL
usage. This prevents a single large call from blowing far past the cap.
"""

from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger

from app.config import DREAM_DAILY_TOKEN_BUDGET
from app.store import DREAM_CHECKPOINTS, QdrantStore

# Fixed Qdrant point ID for the budget ledger (one row, overwritten each update).
_BUDGET_POINT_ID = "00000000-0000-0000-0000-0000000b0d61"  # "budget" mnemonic

# Conservative estimate of prompt-side tokens per dream call when we don't yet
# know the real count (used only for the pre-flight reservation check).
_PROMPT_TOKEN_ESTIMATE = 700


def _utc_day() -> str:
    """Current UTC date as 'YYYY-MM-DD' — the budget reset key."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class DreamTokenBudget:
    """
    Per-UTC-day token budget for the Dream Engine.

    One instance is created at startup and shared by all cycles. It is loaded
    from Qdrant on init so a restart mid-day resumes the same ledger.
    """

    def __init__(self, store: QdrantStore, daily_budget: int = DREAM_DAILY_TOKEN_BUDGET) -> None:
        self._store        = store
        self._daily_budget = daily_budget
        self._day          = _utc_day()
        self._spent        = 0

    async def load(self) -> None:
        """Restore today's spend from Qdrant. Call once at startup."""
        try:
            point = await self._store.get_point(DREAM_CHECKPOINTS, _BUDGET_POINT_ID)
            if point:
                payload = point["payload"]
                if payload.get("day") == self._day:
                    self._spent = int(payload.get("spent", 0))
                    logger.info(
                        f"DreamTokenBudget: resumed — {self._spent}/{self._daily_budget} "
                        f"tokens already spent today ({self._day})"
                    )
                    return
            logger.info(
                f"DreamTokenBudget: fresh day {self._day} — "
                f"budget {self._daily_budget} tokens"
            )
        except Exception as exc:
            logger.warning(f"DreamTokenBudget.load failed (non-fatal): {exc}")

    def _roll_day_if_needed(self) -> None:
        """Reset the counter when the UTC date changes."""
        today = _utc_day()
        if today != self._day:
            logger.info(
                f"DreamTokenBudget: UTC day rolled {self._day} → {today} — "
                f"resetting dream token budget"
            )
            self._day   = today
            self._spent = 0

    @property
    def remaining(self) -> int:
        self._roll_day_if_needed()
        return max(0, self._daily_budget - self._spent)

    def can_afford(self, max_completion_tokens: int) -> bool:
        """
        Pre-flight check: is there enough budget left for one more call,
        assuming the worst case (full completion + estimated prompt)?
        """
        self._roll_day_if_needed()
        estimated_cost = max_completion_tokens + _PROMPT_TOKEN_ESTIMATE
        return self._spent + estimated_cost <= self._daily_budget

    async def record(self, tokens_used: int) -> None:
        """Record ACTUAL tokens consumed by a completed call and persist."""
        self._roll_day_if_needed()
        self._spent += max(0, tokens_used)
        try:
            await self._store.upsert(
                DREAM_CHECKPOINTS,
                [{
                    "id":      _BUDGET_POINT_ID,
                    "vector":  self._store.dummy_vector(),
                    "payload": {
                        "kind":   "dream_token_budget",
                        "day":    self._day,
                        "spent":  self._spent,
                        "budget": self._daily_budget,
                    },
                }],
            )
        except Exception as exc:
            logger.warning(f"DreamTokenBudget.record persist failed (non-fatal): {exc}")

    def status(self) -> str:
        return f"{self._spent}/{self._daily_budget} tokens used today ({self.remaining} left)"
