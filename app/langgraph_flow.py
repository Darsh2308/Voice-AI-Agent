"""
Phase 8 – LangGraph Agent Flow  (enhanced with streaming, tools, summarization)
================================================================================

NEW FEATURES IN THIS FILE
──────────────────────────
Feature 3  – Streaming TTS
  stream_agent() is an async generator that streams tokens from Groq and
  yields complete sentences one-by-one. GroqLangGraphProcessor feeds each
  sentence into TTS immediately so the first audio starts before the LLM
  finishes generating the full response.

Feature 8  – Conversation Summarization
  After every 20 turns the oldest messages are summarized into a single
  system note. Only the summary + last 4 messages are sent to the LLM,
  keeping context within token limits and reducing API costs.

Feature 9  – Tool Calling (Web Search)
  The LLM is given a web_search tool backed by DuckDuckGo (no API key needed).
  If the user asks about current events, weather, or facts the LLM calls the
  tool automatically. The tool result is added to the prompt and the LLM
  produces a final answer.

Feature 10 – Emotion / Tone Detection
  stream_agent() accepts an emotion_hint parameter ("neutral", "hesitant",
  "agitated") and appends a short phrase to the system prompt so the AI
  adapts its tone accordingly.

GRAPH STRUCTURE (unchanged — complexity lives in stream_agent, not the graph)
──────────────────────────────────────────────────────────────────────────────
  START → llm_node → END

Memory is still stored per-thread in MemorySaver.
stream_agent() loads + saves state manually via aget_state / aupdate_state,
which is necessary because ainvoke() would re-run the LLM (defeating streaming).
"""

import json
import re
from typing import Annotated, AsyncGenerator, List

from groq import AsyncGroq
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger
from typing_extensions import TypedDict

from app.config import GROQ_API_KEY
from app.memory import checkpointer


# ─────────────────────────────────────────────────────────────────────────────
# Groq async client (shared across calls)
# ─────────────────────────────────────────────────────────────────────────────

_groq = AsyncGroq(api_key=GROQ_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# State Schema
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    State that flows through the LangGraph nodes.

    messages    – full conversation history (add_messages reducer appends).
    output      – AI's latest reply as plain string (read by run_agent).
    turn_count  – incremented each turn; triggers summarization every 20 turns.
    summary     – rolling conversation summary used when history gets long.
    """
    messages:   Annotated[List[BaseMessage], add_messages]
    output:     str
    turn_count: int
    summary:    str


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt + Emotion Addenda  (Feature 10)
# ─────────────────────────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Keep ALL responses under 2-3 short sentences — you are speaking aloud, not writing. "
    "Never use bullet points, headers, or markdown. "
    "You have memory of the full conversation, so use context from earlier turns. "
    "You have access to a web_search tool — use it when asked about current events, "
    "weather, news, sports scores, or facts you may not know."
)

# Feature 10: tone adjustments appended to the base prompt
EMOTION_ADDENDA = {
    "hesitant": " The user seems uncertain or unclear — be extra clear, patient, and encouraging.",
    "agitated": " The user sounds frustrated or stressed — respond with a calm, brief, empathetic tone.",
    "neutral":  "",
}


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definitions  (Feature 9)
# ─────────────────────────────────────────────────────────────────────────────

# Tool schema passed to the Groq API
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. Use this for questions about "
                "today's weather, news headlines, sports scores, recent events, or any "
                "factual information that may have changed after your training cutoff."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A concise search query (e.g. 'weather Mumbai today')",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


async def _run_web_search(query: str) -> str:
    """
    Feature 9: Execute a DuckDuckGo search and return a brief text summary.
    Uses ddgs (free, no API key required).
    Returns top-3 results joined as a plain string.
    """
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found for that query."
        parts = [f"{r.get('title', '')}: {r.get('body', '')}" for r in results]
        summary = "\n".join(parts)
        logger.info(f"WebSearch({query!r}) → {len(results)} results")
        return summary[:1000]  # trim to avoid huge context
    except Exception as e:
        logger.error(f"WebSearch error: {e}")
        return f"Search failed: {e}"


async def _execute_tool(name: str, arguments_str: str) -> str:
    """Dispatch a tool call by name. Returns the tool result as a string."""
    try:
        args = json.loads(arguments_str)
    except json.JSONDecodeError:
        args = {}

    if name == "web_search":
        return await _run_web_search(args.get("query", ""))
    return f"Unknown tool: {name}"


# ─────────────────────────────────────────────────────────────────────────────
# Sentence Splitting Helper  (Feature 3)
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_MARKUP_RE = re.compile(
    r'<function[^>]*>.*?</function>',
    re.DOTALL | re.IGNORECASE,
)

def _strip_tool_markup(text: str) -> str:
    """Remove any leaked function-call XML that llama sometimes emits as plain text."""
    return _TOOL_MARKUP_RE.sub("", text).strip()


def _extract_sentences(text: str):
    """
    Split text at sentence boundaries (.  ?  !) followed by whitespace or end.
    Returns: (list_of_complete_sentences, remainder_without_ending_punct)

    Examples:
      "Hello! How are you? I'm fine."  → (["Hello!", "How are you?", "I'm fine."], "")
      "Hello! How are"                 → (["Hello!"], "How are")
    """
    # Split on punctuation followed by whitespace OR end of string
    parts = re.split(r'(?<=[.?!])(?:\s+|$)', text)
    if not parts:
        return [], text

    # If last part ends with sentence-ending punct, it's a complete sentence
    if parts[-1] and parts[-1][-1] in ".?!":
        return [p for p in parts if p.strip()], ""

    # Otherwise last part is incomplete
    complete  = [p for p in parts[:-1] if p.strip()]
    remainder = parts[-1]
    return complete, remainder


# ─────────────────────────────────────────────────────────────────────────────
# Conversation Summarization Helper  (Feature 8)
# ─────────────────────────────────────────────────────────────────────────────

async def _summarize_history(config: dict):
    """
    Feature 8: After 20 turns, compress old conversation history.

    Strategy:
      - Keep the last 4 messages intact (for coherence in the next turn).
      - Summarize all older messages into a single paragraph using the LLM.
      - Store the summary in state.summary.
      - The summary is prepended to the system prompt in future turns so the
        LLM retains long-term context without the full message list.

    Note: We do NOT delete messages from state because the add_messages reducer
    only appends. Instead we store the summary separately and cap the messages
    we send to the API at the last 4 (see stream_agent logic below).
    """
    snapshot = await agent_graph.aget_state(config)
    if not snapshot or not snapshot.values:
        return

    messages: List[BaseMessage] = snapshot.values.get("messages", [])
    if len(messages) <= 4:
        return   # nothing old enough to summarize

    old_msgs   = messages[:-4]
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in old_msgs
        if isinstance(m, (HumanMessage, AIMessage))
    )

    logger.info(f"Summarizing {len(old_msgs)} old messages…")
    resp = await _groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": (
                "Summarize this conversation in 3-4 sentences, capturing the key "
                f"topics, decisions, and context:\n\n{history_text}"
            )
        }],
        max_tokens=200,
        temperature=0.3,
    )
    summary = resp.choices[0].message.content or ""
    logger.info(f"Summary generated: {summary[:80]!r}…")

    await agent_graph.aupdate_state(config, {"summary": summary}, as_node="llm")


# ─────────────────────────────────────────────────────────────────────────────
# State Save Helper
# ─────────────────────────────────────────────────────────────────────────────

async def _save_turn(config: dict, user_text: str, ai_text: str, new_turn_count: int):
    """
    Persist the latest user+AI message pair to LangGraph MemorySaver and
    increment the turn counter. Triggers summarization every 20 turns.

    as_node="llm" is required: LangGraph needs to know which node made the
    update so it can determine the next edge (llm → END in our graph).
    Without it, LangGraph raises "Ambiguous update, specify as_node".
    """
    await agent_graph.aupdate_state(
        config,
        {
            "messages":   [HumanMessage(content=user_text), AIMessage(content=ai_text)],
            "output":     ai_text,
            "turn_count": new_turn_count,
        },
        as_node="llm",
    )

    # Feature 8: summarize every 20 turns to keep context manageable
    if new_turn_count > 0 and new_turn_count % 20 == 0:
        await _summarize_history(config)


# ─────────────────────────────────────────────────────────────────────────────
# Graph Node: llm_node  (used by run_agent — kept for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

async def llm_node(state: AgentState) -> dict:
    """
    Single-node LangGraph reasoning function.
    Used by run_agent() (non-streaming fallback).
    stream_agent() bypasses this node and manages state manually.
    """
    summary = state.get("summary", "")
    messages: List[BaseMessage] = state.get("messages", [])

    system = BASE_SYSTEM_PROMPT
    if summary:
        system += f"\n\n[Earlier conversation summary]: {summary}"

    api_messages = [{"role": "system", "content": system}]

    # Feature 8: only send last 4 messages when a summary exists
    visible_msgs = messages[-4:] if summary and len(messages) > 4 else messages
    for msg in visible_msgs:
        if isinstance(msg, HumanMessage):
            api_messages.append({"role": "user",      "content": msg.content})
        elif isinstance(msg, AIMessage):
            api_messages.append({"role": "assistant", "content": str(msg.content)})

    logger.debug(f"llm_node: {len(api_messages)} messages in context")

    resp = await _groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=api_messages,
        temperature=0.7,
        max_tokens=120,
    )
    ai_text = resp.choices[0].message.content.strip()
    logger.info(f"LangGraph llm_node: AI reply → {ai_text!r}")

    return {
        "messages":   [AIMessage(content=ai_text)],
        "output":     ai_text,
        "turn_count": state.get("turn_count", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Graph Assembly  (compiled once at module import)
# ─────────────────────────────────────────────────────────────────────────────

_builder = StateGraph(AgentState)
_builder.add_node("llm", llm_node)
_builder.add_edge(START, "llm")
_builder.add_edge("llm", END)

agent_graph = _builder.compile(checkpointer=checkpointer)
logger.info("LangGraph: agent graph compiled and ready")


# ─────────────────────────────────────────────────────────────────────────────
# Public API: stream_agent()  — Feature 3 primary entry point
# ─────────────────────────────────────────────────────────────────────────────

async def stream_agent(
    user_text: str,
    thread_id: str,
    emotion_hint: str = "neutral",
) -> AsyncGenerator[str, None]:
    """
    Feature 3 (Streaming TTS): Async generator that yields complete sentences
    from the LLM response one-by-one as they are generated.

    Flow:
      1. Load conversation history from LangGraph MemorySaver.
      2. Build the API message list (with optional summary for Feature 8).
      3. Make a non-streaming call with tool support (Feature 9).
         If the model returns a tool_call → execute it → make a second call.
         If no tool call → yield sentences from the first response directly.
      4. After all sentences are yielded, persist the full response to state.
      5. Trigger summarization if turn_count % 20 == 0 (Feature 8).

    Why two API calls for tool queries?
      Groq streaming with tools requires assembling the full tool-call JSON
      from many tiny deltas, which adds fragile parsing complexity. The simpler
      approach: one non-streaming call detects the tool call fast (~200 ms),
      then a second streaming call produces the final user-facing response.
      For 90%+ of turns (no tools), there is only ONE non-streaming call and
      sentences are yielded directly from its content — same latency as before.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # ── 1. Load history ───────────────────────────────────────────────────────
    snapshot = await agent_graph.aget_state(config)
    if snapshot and snapshot.values:
        existing_msgs: List[BaseMessage] = snapshot.values.get("messages", [])
        turn_count: int                  = snapshot.values.get("turn_count", 0)
        summary: str                     = snapshot.values.get("summary", "")
    else:
        existing_msgs = []
        turn_count    = 0
        summary       = ""

    # ── 2. Build system prompt ────────────────────────────────────────────────
    # Feature 10: append emotion-based tone adjustment
    system = BASE_SYSTEM_PROMPT + EMOTION_ADDENDA.get(emotion_hint, "")
    if summary:
        # Feature 8: prepend summary when history has been compressed
        system += f"\n\n[Earlier conversation summary]: {summary}"

    api_messages = [{"role": "system", "content": system}]

    # Feature 8: only send last 4 messages when summary is present
    visible_msgs = existing_msgs[-4:] if summary and len(existing_msgs) > 4 else existing_msgs
    for msg in visible_msgs:
        if isinstance(msg, HumanMessage):
            api_messages.append({"role": "user",      "content": msg.content})
        elif isinstance(msg, AIMessage):
            api_messages.append({"role": "assistant", "content": str(msg.content)})

    # Add the current user message
    api_messages.append({"role": "user", "content": user_text})

    logger.debug(f"stream_agent: {len(api_messages)} messages, emotion={emotion_hint!r}, summary={'yes' if summary else 'no'}")

    # ── 3. First call: detect tool usage (Feature 9) ─────────────────────────
    first_resp = await _groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=api_messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=200,
        temperature=0.7,
    )

    first_choice    = first_resp.choices[0]
    finish_reason   = first_choice.finish_reason
    full_response   = ""

    if finish_reason == "tool_calls":
        # ── Feature 9: Execute tool(s) and get augmented response ────────────
        tool_calls = first_choice.message.tool_calls or []
        logger.info(f"stream_agent: tool_calls detected — {[tc.function.name for tc in tool_calls]}")

        # Tell the user we're searching (immediate sentence so TTS doesn't stall)
        yield "Let me look that up for you."

        # Build the tool result messages
        tool_results = []
        for tc in tool_calls:
            result = await _execute_tool(tc.function.name, tc.function.arguments)
            tool_results.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

        # Reconstruct the assistant tool-call message for the follow-up request
        assistant_tool_msg = {
            "role":       "assistant",
            "tool_calls": [
                {
                    "id":   tc.id,
                    "type": "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        }

        follow_up_messages = api_messages + [assistant_tool_msg] + tool_results

        # Stream the final answer after tool execution
        stream = await _groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=follow_up_messages,
            max_tokens=150,
            temperature=0.7,
            stream=True,
        )
        buffer = ""
        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            full_response += token
            buffer        += token
            sentences, buffer = _extract_sentences(buffer)
            for s in sentences:
                clean = _strip_tool_markup(s)
                if clean:
                    yield clean

        if buffer.strip():
            clean = _strip_tool_markup(buffer)
            if clean:
                yield clean
            full_response = full_response  # already accumulated

    else:
        # ── No tool call: yield sentences from the first response ─────────────
        content = _strip_tool_markup(first_choice.message.content or "").strip()
        full_response = content

        if not content:
            logger.warning("stream_agent: empty LLM response")
            return

        # Split into sentences and yield each
        sentences, remainder = _extract_sentences(content + " ")
        for s in sentences:
            clean = _strip_tool_markup(s)
            if clean:
                yield clean
        if remainder.strip():
            clean = _strip_tool_markup(remainder)
            if clean:
                yield clean

    # ── 4. Save state ─────────────────────────────────────────────────────────
    # Wrapped in try/except so a state-save failure doesn't propagate as an
    # exception through the async generator into _generate(), which would have
    # suppressed the TranscriptDisplayFrame before the `finally` fix was added.
    if full_response.strip():
        try:
            await _save_turn(config, user_text, full_response.strip(), turn_count + 1)
        except Exception as e:
            logger.error(f"stream_agent: state save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Public API: run_agent()  — non-streaming fallback (backward compat)
# ─────────────────────────────────────────────────────────────────────────────

async def run_agent(user_input: str, thread_id: str) -> str:
    """
    Non-streaming agent call. Still used for testing and the POST /voice endpoint.
    Internally uses the LangGraph ainvoke path through llm_node.
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
    Returns a list of dicts: [{"role": "user"/"assistant", "text": "..."}]
    """
    config = {"configurable": {"thread_id": thread_id}}
    state  = await agent_graph.aget_state(config)
    if not state or not state.values:
        return []
    history = []
    for msg in state.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            history.append({"role": "user",      "text": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "text": msg.content})
    return history
