"""
Phase 8 – LangGraph Agent Flow
================================

WHAT IS LANGGRAPH?
──────────────────
LangGraph is a library for building stateful, multi-step AI agents as
directed graphs. Each node in the graph is a function (or coroutine) that:
  1. Reads from the current state
  2. Does some work (call LLM, call a tool, make a decision, …)
  3. Returns partial state updates

The graph "reducer" merges those updates back into the full state,
so the next node gets a complete, up-to-date view of everything.

WHY LANGGRAPH OVER A PLAIN LIST?
──────────────────────────────────
Phase 7's GroqLLMProcessor maintained this list manually:
    self._messages.append({"role": "user",  "content": user_text})
    self._messages.append({"role": "assistant", "content": ai_reply})

That's fine for Phase 7, but it has no structure beyond a list.
LangGraph gives you:

  ┌────────────────────────────────────────────────────────────────┐
  │  PLAIN LIST (Phase 7)       │  LANGGRAPH STATE (Phase 8)       │
  │─────────────────────────────┼──────────────────────────────────│
  │  Manual append/pop          │  Automatic via add_messages       │
  │  No rollback safety         │  Checkpointed — never corrupts    │
  │  dict objects               │  Typed HumanMessage/AIMessage     │
  │  No hook points             │  Add tools/routing easily         │
  │  Can't inspect mid-stream   │  Full state visible at each step  │
  └────────────────────────────────────────────────────────────────┘

OUR GRAPH (Phase 8)
────────────────────
Phase 8 uses the simplest possible graph: one node.
This is the right starting point — it's easy to add more nodes later.

    START
      │
      ▼
  ┌──────────┐
  │ llm_node │  Reads full message history from state,
  │          │  calls Groq, adds AI reply to state.
  └──────────┘
      │
      ▼
     END

STATE SCHEMA  (AgentState)
────────────────────────────
    messages: Annotated[List[BaseMessage], add_messages]
              ↑ The `add_messages` reducer means:
                "when a node returns {'messages': [new_msg]},
                 APPEND it to the existing list, don't replace it."
              This is how memory accumulates across turns automatically.

    output:   str   ← the AI's latest reply, read by GroqLangGraphProcessor

HOW MEMORY WORKS ACROSS TURNS
──────────────────────────────
Turn 1 — ainvoke({"messages": [HumanMessage("Hi")]}, thread_id="abc")
  State before node: messages = [HumanMessage("Hi")]
  Node returns:      {"messages": [AIMessage("Hello!")], "output": "Hello!"}
  State after node:  messages = [HumanMessage("Hi"), AIMessage("Hello!")]
  Saved in MemorySaver under thread_id "abc"

Turn 2 — ainvoke({"messages": [HumanMessage("My name is DSP")]}, thread_id="abc")
  LangGraph loads previous state from MemorySaver
  State before node: messages = [HumanMessage("Hi"), AIMessage("Hello!"),
                                  HumanMessage("My name is DSP")]
  → Full history is visible to the LLM!
  Node returns:      {"messages": [AIMessage("Got it, DSP!")], "output": "Got it, DSP!"}
  State after node:  messages = [Hi, Hello!, My name is DSP, Got it DSP!]
  Saved back to MemorySaver

Turn 3 — ainvoke({"messages": [HumanMessage("What is my name?")]}, thread_id="abc")
  LangGraph loads saved state — ALL 4 messages are there
  LLM sees full history → responds "Your name is DSP"  ✓
"""

from typing import Annotated, List

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger
from typing_extensions import TypedDict

from app.config import GROQ_API_KEY
from app.memory import checkpointer


# ─────────────────────────────────────────────────────────────────────────────
# State Schema
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    The state object that flows through the LangGraph nodes.

    messages
    ────────
    A list of LangChain message objects (HumanMessage, AIMessage, etc.).
    The `add_messages` reducer means that when any node returns
    {"messages": [some_new_message]}, LangGraph APPENDS it to the existing
    list instead of replacing it.

    This single annotation is what gives the AI its cross-turn memory —
    the list grows with every turn and the full history is always in state.

    output
    ──────
    The AI's latest response as a plain string.
    GroqLangGraphProcessor reads this after each ainvoke() call.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    output: str


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Keep ALL responses under 2-3 short sentences — you are speaking aloud, not writing. "
    "Never use bullet points, headers, or markdown. "
    "You have memory of the full conversation, so use context from earlier turns."
)


# ─────────────────────────────────────────────────────────────────────────────
# Graph Node: llm_node
# ─────────────────────────────────────────────────────────────────────────────

async def llm_node(state: AgentState) -> dict:
    """
    The single reasoning node of our LangGraph agent.

    Receives the full state (all messages so far + current user message),
    builds the API call, calls Groq, and returns the AI's reply.

    Returns a dict of state UPDATES (not the full state):
      - "messages": [AIMessage(...)]  → gets APPENDED to state.messages
      - "output":   str               → gets SET in state.output

    After this node, LangGraph:
      1. Applies the add_messages reducer → history grows by 1 AI message
      2. Saves the updated state in MemorySaver under the current thread_id
      3. Returns control to the caller (run_agent)
    """

    # ── Build the API message list ────────────────────────────────────────────
    # Convert LangChain message objects to the dict format Groq's API expects.
    # state["messages"] already contains the full history including the user's
    # latest message — LangGraph added it before calling this node.
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            api_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            api_messages.append({"role": "assistant", "content": msg.content})
        # SystemMessage, ToolMessage, etc. are ignored for now

    logger.debug(f"LangGraph llm_node: {len(api_messages)} messages in context")

    # ── Call Groq API ─────────────────────────────────────────────────────────
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": api_messages,
                "temperature": 0.7,
                "max_tokens": 120,
            },
        )
        resp.raise_for_status()

    ai_text = resp.json()["choices"][0]["message"]["content"].strip()
    logger.info(f"LangGraph: AI reply → {ai_text!r}")

    # ── Return state updates ──────────────────────────────────────────────────
    # Returning {"messages": [AIMessage(...)]} triggers the add_messages reducer:
    # the AI's reply gets APPENDED to state.messages, not replacing the whole list.
    return {
        "messages": [AIMessage(content=ai_text)],
        "output": ai_text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Graph Assembly  (runs once at module import time)
# ─────────────────────────────────────────────────────────────────────────────
#
# StateGraph(AgentState) creates a graph whose nodes share the AgentState schema.
# add_node("llm", llm_node) registers our coroutine as the "llm" node.
# add_edge(START, "llm")    means: first thing to run is "llm".
# add_edge("llm", END)      means: after "llm", we're done.
#
# .compile(checkpointer=checkpointer) wires in the MemorySaver:
# every ainvoke() will now automatically load + save state.
#
# We build the graph ONCE at module level so it's shared across all connections.
# The per-connection isolation comes from thread_id, not from separate graph instances.
#

_builder = StateGraph(AgentState)
_builder.add_node("llm", llm_node)
_builder.add_edge(START, "llm")
_builder.add_edge("llm", END)

agent_graph = _builder.compile(checkpointer=checkpointer)

logger.info("LangGraph: agent graph compiled and ready")


# ─────────────────────────────────────────────────────────────────────────────
# Public API: run_agent()
# ─────────────────────────────────────────────────────────────────────────────

async def run_agent(user_input: str, thread_id: str) -> str:
    """
    Run one turn of the conversation agent.

    Parameters
    ──────────
    user_input : the user's transcribed speech for this turn
    thread_id  : UUID string identifying this WebSocket session.
                 LangGraph uses this to load/save the right conversation.

    How it works
    ────────────
    1. LangGraph loads the state for `thread_id` from MemorySaver.
       (On first call: empty state — no messages yet.)

    2. The input {"messages": [HumanMessage(user_input)]} is MERGED into state
       via the add_messages reducer → user's message is appended to history.

    3. llm_node runs with the full state (all previous messages + new one).

    4. llm_node returns {"messages": [AIMessage(reply)], "output": reply}.
       LangGraph appends the AI message to history and saves state.

    5. We return result["output"] — just the AI's text reply.

    Returns
    ───────
    The AI's response as a plain string.
    """
    result = await agent_graph.ainvoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result.get("output", "")


# ─────────────────────────────────────────────────────────────────────────────
# Utility: get_conversation_history()
# ─────────────────────────────────────────────────────────────────────────────

async def get_conversation_history(thread_id: str) -> list:
    """
    Retrieve the full message history for a session.
    Useful for debugging, display, or export.

    Returns a list of dicts: [{"role": "user"/"assistant", "text": "..."}]
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = await agent_graph.aget_state(config)

    if not state or not state.values:
        return []

    history = []
    for msg in state.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "text": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "text": msg.content})

    return history
