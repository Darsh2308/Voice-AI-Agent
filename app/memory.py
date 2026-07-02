"""
Phase 8 – Memory Infrastructure
=================================

WHY THIS FILE EXISTS
────────────────────
Phase 7's GroqLLMProcessor kept memory as a plain Python list:
    self._messages = [{"role": "system", ...}, {"role": "user", ...}, ...]

This works, but it's just a list. There's no:
  - Formal state management
  - Ability to inspect / replay history
  - Hook points for tools, routing, or persistence
  - Thread isolation (if we wanted multiple users)

Phase 8 upgrades memory using two LangGraph/LangChain primitives:

  1. MemorySaver  (LangGraph built-in)
     ────────────────────────────────
     LangGraph's checkpoint store. Think of it as a dictionary:
         { thread_id → full conversation state snapshot }

     When you call agent_graph.ainvoke(..., config={"configurable":{"thread_id": "abc"}})
     LangGraph automatically:
       - loads the previous state for "abc"
       - runs the graph nodes
       - saves the updated state back under "abc"

     You get persistent memory across turns with ZERO manual bookkeeping.

  2. ConversationBufferMemory  (LangChain — shown for reference)
     ────────────────────────────────────────────────────────────
     The classic LangChain memory object. Stores pairs of:
         human: "My name is DSP"
         ai:    "Nice to meet you, DSP!"

     Exposed here for reference and for cases where you want to export
     history as plain text (e.g. for debugging, logging, or display).

MEMORY SCOPE
────────────
  MemorySaver stores everything IN RAM.
  - It resets when the server restarts.
  - Each unique thread_id is an isolated conversation.
  - One MemorySaver instance is shared across ALL connections — isolation
    is enforced by the thread_id, not by having separate objects.

  ┌─────────────────────────────────────────────────────────────┐
  │  MemorySaver (single global instance)                       │
  │                                                             │
  │  thread_id: "abc123" → [Human: "Hi", AI: "Hello!", …]      │
  │  thread_id: "def456" → [Human: "Bye", AI: "Goodbye!", …]   │
  │  …                                                          │
  └─────────────────────────────────────────────────────────────┘

  Each WebSocket connection generates a new UUID as its thread_id,
  so every session is completely isolated.

FUTURE UPGRADES (optional next steps)
──────────────────────────────────────
  Replace MemorySaver with:
  - SqliteSaver  → persists to a local SQLite file (survives restarts)
  - RedisSaver   → persists to Redis (works across multiple servers)
  - PostgresSaver → persists to PostgreSQL (production scale)
  All have the same interface; just swap the checkpointer.
"""

from langgraph.checkpoint.memory import MemorySaver

# ── Shared in-RAM checkpointer ────────────────────────────────────────────────
#
# ONE instance for the whole server process.
# Thread safety: MemorySaver is coroutine-safe (uses asyncio-compatible writes).
# Isolation: guaranteed by thread_id — different connections never see each other.
#
checkpointer = MemorySaver()
