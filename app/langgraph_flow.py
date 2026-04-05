"""
Phase 8 – LangGraph Agent Flow  (enhanced with streaming, tools, summarization)
================================================================================

NEW FEATURES IN THIS FILE
──────────────────────────
Feature 3  – Streaming TTS (word-chunk flushing)
  stream_agent() is an async generator that opens a single streaming call to
  Groq and yields FLUSH_WORD_COUNT-word chunks as tokens arrive — no waiting
  for a sentence boundary. GroqLangGraphProcessor feeds each chunk into TTS
  immediately, so the first audio byte reaches the browser within ~200 ms of
  the user finishing speech (down from ~400 ms with sentence flushing).

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

# BCP-47 → human-readable name for the language instruction injected into the system prompt
LANG_NAMES: dict[str, str] = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "mr-IN": "Marathi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "kn-IN": "Kannada",
    "bn-IN": "Bengali",
    "gu-IN": "Gujarati",
    "pa-IN": "Punjabi",
    "ml-IN": "Malayalam",
    "or-IN": "Odia",
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


FLUSH_WORD_COUNT = 5  # emit a TTS chunk every N words

def _flush_words(buffer: str, n: int = FLUSH_WORD_COUNT):
    """
    Extract complete n-word chunks from a streaming token buffer.

    A word is 'complete' when followed by whitespace. The last word in the
    buffer is kept as a partial (it may still be mid-token from the LLM stream)
    unless the buffer ends with whitespace, in which case all words are complete.

    Returns: (list_of_n_word_chunks, remaining_buffer)

    Examples:
      "Hello how are you today " (trailing space) → (["Hello how are you today"], "")
      "Hello how are you today"  (no trail space) → ([], "Hello how are you today")
      "one two three four five six " (6 words)   → (["one two three four five"], "six ")
    """
    trailing_space = bool(buffer) and buffer[-1].isspace()
    words    = buffer.split()
    complete = words if trailing_space else words[:-1]
    partial  = "" if trailing_space else (words[-1] if words else "")

    chunks = []
    while len(complete) >= n:
        chunks.append(" ".join(complete[:n]))
        complete = complete[n:]

    # Rebuild remaining buffer preserving trailing-space signal for next call
    remaining = " ".join(complete)
    if remaining and partial:
        remaining += " " + partial
    elif partial:
        remaining = partial
    elif remaining:
        remaining += " "   # preserve trailing space so next call sees complete words

    return chunks, remaining


def _flush_all(buffer: str) -> str:
    """Return the full buffer stripped — used to flush the final fragment."""
    return buffer.strip()


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
    language: str = "en-IN",
) -> AsyncGenerator[str, None]:
    """
    Feature 3 (Streaming TTS): Async generator that yields FLUSH_WORD_COUNT-word
    chunks from the LLM token stream in real-time — no waiting for a full sentence.

    Flow:
      1. Load conversation history from LangGraph MemorySaver.
      2. Build the API message list (with optional summary for Feature 8).
      3. Open a SINGLE streaming call with tool support (Feature 9).
         - Content tokens → accumulated in word_buffer, flushed every FLUSH_WORD_COUNT words.
         - Tool-call deltas → assembled from streaming deltas (index-keyed accumulator).
      4. If tool call detected:
         - Yield "Let me look that up for you." immediately so TTS doesn't stall.
         - Execute tool(s), stream the follow-up answer with the same word flushing.
      5. After all chunks are yielded, persist full response to LangGraph state.

    Why one streaming call instead of the old two-call design?
      The previous approach used a non-streaming first call to detect tool use,
      then a second streaming call for the actual answer. That meant the user
      waited for a full round-trip (~200-400 ms) before the first audio byte,
      even for the 90 %+ of turns that never use tools.
      Now a single streaming call starts delivering tokens immediately; if tool
      deltas appear we handle them without an extra round-trip.
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
    system = BASE_SYSTEM_PROMPT + EMOTION_ADDENDA.get(emotion_hint, "")

    # Language instruction: always tell the LLM which language to reply in.
    # The detected language comes from Sarvam STT on every turn.
    lang_name = LANG_NAMES.get(language, language)
    system += (
        f" IMPORTANT: The user is speaking in {lang_name}. "
        f"You MUST reply ONLY in {lang_name}. "
        f"Do not translate to English or switch languages under any circumstances."
    )

    if summary:
        system += f"\n\n[Earlier conversation summary]: {summary}"

    api_messages = [{"role": "system", "content": system}]

    visible_msgs = existing_msgs[-4:] if summary and len(existing_msgs) > 4 else existing_msgs
    for msg in visible_msgs:
        if isinstance(msg, HumanMessage):
            api_messages.append({"role": "user",      "content": msg.content})
        elif isinstance(msg, AIMessage):
            api_messages.append({"role": "assistant", "content": str(msg.content)})

    api_messages.append({"role": "user", "content": user_text})

    logger.debug(f"stream_agent: {len(api_messages)} messages, emotion={emotion_hint!r}, summary={'yes' if summary else 'no'}")

    # ── 3. Single streaming call — handles both content and tool-call paths ───
    stream = await _groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=api_messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=300,
        temperature=0.7,
        stream=True,
    )

    word_buffer:    str  = ""
    full_response:  str  = ""
    is_tool_call:   bool = False
    # Accumulate tool-call JSON from streaming deltas (keyed by delta index)
    tool_calls_acc: dict = {}

    async for chunk in stream:
        choice       = chunk.choices[0]
        delta        = choice.delta

        # ── Tool-call delta: accumulate function name + arguments ─────────────
        if delta.tool_calls:
            is_tool_call = True
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {
                        "id":       "",
                        "type":     "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc_delta.id:
                    tool_calls_acc[idx]["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tool_calls_acc[idx]["function"]["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        tool_calls_acc[idx]["function"]["arguments"] += tc_delta.function.arguments

        # ── Content token: flush every FLUSH_WORD_COUNT words ─────────────────
        elif delta.content:
            token         = delta.content
            full_response += token
            word_buffer   += token
            chunks, word_buffer = _flush_words(word_buffer)
            for c in chunks:
                clean = _strip_tool_markup(c)
                if clean.strip():
                    yield clean.strip()

    # Flush any remaining words after the stream ends (no-tool path)
    if not is_tool_call and word_buffer.strip():
        clean = _strip_tool_markup(_flush_all(word_buffer))
        if clean:
            yield clean

    # ── 4. Tool-call path: execute tool(s) then stream the follow-up ─────────
    if is_tool_call:
        tool_call_list = list(tool_calls_acc.values())
        logger.info(f"stream_agent: tool_calls detected — {[tc['function']['name'] for tc in tool_call_list]}")

        # Yield a bridging phrase immediately so TTS doesn't go silent
        yield "Let me look that up for you."

        tool_results = []
        for tc in tool_call_list:
            result = await _execute_tool(tc["function"]["name"], tc["function"]["arguments"])
            tool_results.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "content":      result,
            })

        assistant_tool_msg = {
            "role":       "assistant",
            "tool_calls": [
                {
                    "id":   tc["id"],
                    "type": "function",
                    "function": {
                        "name":      tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for tc in tool_call_list
            ],
        }

        follow_up_messages = api_messages + [assistant_tool_msg] + tool_results

        stream2 = await _groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=follow_up_messages,
            max_tokens=200,
            temperature=0.7,
            stream=True,
        )

        word_buffer2   = ""
        full_response2 = ""
        async for chunk in stream2:
            token = chunk.choices[0].delta.content or ""
            if not token:
                continue
            full_response2 += token
            word_buffer2   += token
            chunks, word_buffer2 = _flush_words(word_buffer2)
            for c in chunks:
                clean = _strip_tool_markup(c)
                if clean.strip():
                    yield clean.strip()

        if word_buffer2.strip():
            clean = _strip_tool_markup(_flush_all(word_buffer2))
            if clean:
                yield clean

        full_response = "Let me look that up for you. " + full_response2

    # ── 5. Save state ─────────────────────────────────────────────────────────
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
