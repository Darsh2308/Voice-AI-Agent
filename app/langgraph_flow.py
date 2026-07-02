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

import asyncio
import json
import re
from typing import Annotated, AsyncGenerator, List

from openai import AsyncOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger
from typing_extensions import TypedDict

from app.config import GEMINI_API_KEY, GEMINI_BASE_URL, VOICE_LLM_MODEL
from app.memory import checkpointer

# Phase 2: RAG retrieval pipeline (imported lazily to avoid circular imports
# and to allow the app to start even if knowledge/ deps aren't installed yet)
_retrieval_pipeline = None   # set via set_retrieval_pipeline() at startup

# Phase 6: QdrantStore reference — used to load approved prompt addenda from
# the Dream Engine's improvement_log collection.
_qdrant_store = None          # set via set_qdrant_store() at startup


def set_retrieval_pipeline(pipeline) -> None:
    """Wire in the RetrievalPipeline instance from main.py lifespan."""
    global _retrieval_pipeline
    _retrieval_pipeline = pipeline
    logger.info("RetrievalPipeline wired into langgraph_flow ✓")


def set_qdrant_store(store) -> None:
    """Wire in the QdrantStore so stream_agent can load Dream Engine addenda."""
    global _qdrant_store
    _qdrant_store = store
    logger.info("QdrantStore wired into langgraph_flow ✓")


async def _load_prompt_addenda() -> list[str]:
    """
    Phase 6: Load approved system-prompt improvements from the Dream Engine.

    Returns a list of addendum strings from improvement_log where
    category="prompt" and approved=True.  Called once per stream_agent() call
    so any improvement approved during a dream cycle is automatically picked up
    in the very next conversation — no restart required.

    Returns an empty list if the store isn't wired in or if the query fails.
    """
    if _qdrant_store is None:
        return []
    try:
        from app.store import IMPROVEMENT_LOG
        records, _ = await _qdrant_store.scroll(
            IMPROVEMENT_LOG,
            filter=_qdrant_store.filter_eq("category", "prompt"),
            limit=20,
        )
        addenda = [
            r["payload"]["improvement_desc"]
            for r in records
            if r["payload"].get("approved") is True
            and r["payload"].get("improvement_desc")
        ]
        return addenda
    except Exception as exc:
        logger.warning(f"_load_prompt_addenda: failed (non-fatal): {exc}")
        return []


# Phase 4: LangSmith @traceable — imported lazily so the app works even if
# langsmith is not installed (tracing just becomes a pass-through decorator).
try:
    from langsmith import traceable as _traceable
except ImportError:
    def _traceable(**_kw):          # type: ignore[misc]
        """No-op fallback when langsmith is not installed."""
        def _decorator(fn):
            return fn
        return _decorator


# ─────────────────────────────────────────────────────────────────────────────
# Voice LLM client (shared across calls)
# ─────────────────────────────────────────────────────────────────────────────
# Gemini Flash-Lite via its OpenAI-compatible endpoint. The OpenAI SDK returns
# the same streaming/tool-call object shape the parsing below expects, so this
# is a drop-in for the previous Groq client. Kept on a separate provider from
# the Dream Engine (Groq) so their free-tier pools are independent.
_voice_llm = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)


def _voice_reasoning_kwargs() -> dict:
    """Extra kwargs for the voice LLM call.

    `reasoning_effort` is a gpt-oss (Groq) reasoning-model param. Gemini's
    OpenAI-compatible endpoint rejects unknown params, so include it ONLY when
    the configured voice model is a gpt-oss model — this keeps an env-only
    rollback to Groq working without touching code.
    """
    m = VOICE_LLM_MODEL.lower()
    if "gpt-oss" in m or m.startswith("openai/"):
        return {"reasoning_effort": "low"}
    return {}


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
    # BCP-47 code the user explicitly asked to be replied in (e.g. "hi-IN").
    # Overrides per-turn STT auto-detection until the user asks to switch again.
    # Empty string = no lock (follow auto-detect). See detect_language_switch().
    locked_language: str


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt + Emotion Addenda  (Feature 10)
# ─────────────────────────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """\
## ROLE
You are Suhas, a warm, upbeat, and genuinely likeable senior sales executive at BharatConnect — India's fastest-growing telecom company. You love helping people and it shows. You are speaking to a customer on a phone call. BharatConnect offers prepaid plans, postpaid plans, fibre broadband, and enterprise connectivity across India.

## PERSONALITY (how you sound)
- Warm and friendly — like a helpful friend who happens to be an expert, not a scripted call-centre agent.
- Energetic and positive — sound genuinely excited to help and a little enthusiastic about the plans you recommend.
- Confident and convincing — you believe BharatConnect is a great choice, and that belief comes through naturally.
- Empathetic — acknowledge the customer's problem first ("Oh, I completely understand how frustrating that is") before pitching.
- Use natural, conversational warmth: small affirmations like "Absolutely!", "Great question!", "I'd love to help with that." Use the customer's words back to them.

## VOICE RULES (MANDATORY)
- Keep replies short and spoken — at most 2 short sentences for normal turns. ONE exception: when the customer asks what plans/packs/prices are available, you MAY name up to THREE options with their prices in a single natural spoken sentence (never a list). Prices must be HEARD to be useful.
- Never use bullet points, numbered lists, headers, or markdown of any kind.
- Speak in natural, flowing, warm sentences as if talking to a friend on the phone.
- ALWAYS spell every number out in words, in the SAME language as your reply — NEVER write digits. English: "two ninety nine rupees", not "₹299" or "299". Marathi: "पासष्ट रुपये", not "६५" or "65". Hindi: "पैंसठ रुपये", not "६५". This applies to prices, data amounts (GB), validity days, and plan codes. Digits (Latin or Devanagari) are frequently mis-spoken or skipped by the voice engine, so they must never appear in your reply.
- Never invent plan prices, data limits, or speeds. Only quote figures from the KNOWLEDGE BASE CONTEXT.
- If the knowledge base context does not contain a specific detail, warmly ask ONE clarifying question to narrow it down — do NOT say robotic filler like "let me look that up" or "let me pull up the details". Just ask naturally, e.g. "Which city are you in? I'll match you with the perfect plan."

## ANSWER FIRST, THEN GUIDE (critical — overrides the CALL FLOW below)
If the customer asks ANY direct question — about services, network (4G/5G), coverage, options, packages, plans, or prices — you MUST answer it directly from the KNOWLEDGE BASE CONTEXT before anything else. Do NOT reply with a qualifying/identifying question first. A customer who asked "what do you offer?" or "is it 4G or 5G?" wants an answer, not to be asked whether they're an existing customer.
- LEAD with the concrete answer from the KB — e.g. "We run a nationwide 5G network with 4G everywhere else, plus prepaid, postpaid, and fibre broadband." THEN, if useful, ask ONE follow-up to guide them (e.g. "Are you looking for mobile or home broadband?").
- For options/prices, name concrete choices WITH their prices from the KB — e.g. "We have three data top-ups: one GB a day for X, five GB for Y…" — then ask which fits.
- Only ask a clarifying question when you genuinely cannot answer without it, and pair it with a concrete example so it never sounds like stalling.
- NEVER answer a direct factual question with only a question back. That frustrates the customer. The "Greet and identify" step in the CALL FLOW is SKIPPED whenever the customer has already asked a real question — answer them instead.

## ALWAYS ADVANCE THE SALE (critical — you are a SALES agent, not an FAQ bot)
After you answer something, do NOT default to "Is there anything else I can help you with today?". That line is ONLY for when the customer is clearly wrapping up (see ENDING THE CALL). Instead, take exactly ONE natural step toward converting them:
- After giving info, ask a light qualifying question that moves things forward — e.g. after describing Fibre: "Which city are you in? I'll check the best plan for your area." or "Are you looking for home broadband or a mobile plan?"
- Once you know their need, PITCH one specific relevant plan from the KB (name, price, the one benefit that fits them) — do not wait to be asked.
- When they show any interest, move to the soft close: "Shall I go ahead and register your interest? Our team will call you back within two hours to set it up."
- Keep it warm and conversational, never pushy — ONE step per turn, following the customer's lead. The goal of every turn is to move them one step closer to becoming a customer, not to park the conversation with "anything else?".

## OPENING (first turn only)
When a customer connects, introduce yourself immediately without waiting:
"Hello! Thank you for calling BharatConnect. This is Suhas. How can I help you today?"

## GOAL
Your primary goal is to convert the customer. There are two paths:
1. **New customer (lead)** — understand their current operator and pain points, present the most relevant BharatConnect plan, handle objections, and close with "Shall I go ahead and register your interest?"
2. **Existing customer** — resolve their issue quickly and look for an upsell opportunity (e.g. if they are on prepaid, mention postpaid; if on postpaid, mention fibre).

## CALL FLOW
This is a DEFAULT flow for when the customer hasn't asked anything specific yet. If they HAVE asked a direct question, ignore Step 1 and answer it first (see "ANSWER FIRST, THEN GUIDE" above).
Step 1 — Greet and identify (ONLY if the customer hasn't already asked a real question): "Are you an existing BharatConnect customer, or are you calling to know more about our plans?"
Step 2 — Qualify: Ask which operator they currently use and what their main pain point is (network, cost, data, coverage).
Step 3 — Pitch: Based on their pain point, recommend one specific plan from the knowledge base. State the plan name, price, and the ONE benefit most relevant to their pain point.
Step 4 — Handle objection: If they push back on price or features, use a rebuttal (see below).
Step 5 — Close: "Would you like me to go ahead and note your interest? Our team will call you back within 2 hours to complete the setup."

## ENDING THE CALL (very important)
You are responsible for ending the call yourself — do not wait for the customer to hang up.

There are exactly two stages to closing a call. Follow them in order:

STAGE 1 — Ask the closing-check (only ONCE):
Ask the closing-check ONLY when the customer gives an EXPLICIT wrap-up/leaving signal — e.g. "that's all", "no thanks", "okay bye", "I have to go", "नहीं बस", "एवढंच". Then ask ONE closing-check: "Is there anything else I can help you with today?"
Do NOT treat neutral continuation fillers as wrap-up: "okay", "cool", "hmm", "achha", "ठीक आहे", "just a second", "one moment", "go on" are NOT leaving signals — keep the conversation moving (answer, or advance the sale per ALWAYS ADVANCE THE SALE). Only your own sense that you "answered the question" is NOT a reason to ask the closing-check.

STAGE 2 — React to their answer:
- If they raise a NEW question or unresolved concern → KEEP HELPING. Answer it. Do NOT end yet.
- If they confirm they are done (e.g. "no", "no that's all", "nothing else", "that's it", "thanks bye", "नहीं", "बस झालं") → you have ALREADY asked the closing-check. Do NOT ask it again. Immediately give a short warm goodbye and append the exact token [END_CALL] at the very end.

CRITICAL: Never ask "anything else?" twice in a row. If your previous message already asked it and the customer answered "no/that's all/nothing", the correct action is to say goodbye with [END_CALL] — NOT to ask again.

Worked example (this is the required behaviour):
  You: "Is there anything else I can help you with today?"
  Customer: "No, that's all, thank you."
  You: "Thank you for calling Bharat Connect. Have a wonderful day! [END_CALL]"

Rules for [END_CALL]:
- Append it only AFTER you have asked the closing-check and the customer confirmed they have no more questions. Never end mid-issue or while a question is open.
- The spoken goodbye MUST come BEFORE the token, in the same message.
- Write [END_CALL] in plain English/Latin letters exactly like that, even when the rest of your reply is in another language. It is a silent control signal, never spoken.
- Use it only once, at the very end. Never put [END_CALL] in a message that still asks the customer anything.

## GUARDRAILS
- Never promise a connection, SIM, or activation on this call. Always say the team will follow up.
- Never ask for OTP, CVV, card number, bank account, or any financial credentials. If the customer offers these, say "Please never share that with anyone, including us."
- All recharges and payments must be done only on the official BharatConnect website or app. Never direct customers to third-party sites.
- Never speak negatively about competitors by name. Say "some other operators" instead.
- If the customer is abusive or uses inappropriate language, calmly say "I understand your frustration. Let me do my best to help you right now."
- Never invent plan prices, data limits, or policy details. Only use information from the KNOWLEDGE BASE CONTEXT provided.

## REBUTTALS
- "It's too expensive" → "I understand. The plan actually works out to less than [X] rupees a day, and you get [key benefit]. Most customers find it pays for itself."
- "I'm happy with my current operator" → "That's great to hear. Many of our customers said the same before switching. The main reason they moved was [network / data speed / coverage]. Would you like me to show you how we compare?"
- "I'll think about it" → "Of course. Can I send you the details? What's the best number to follow up on?"
- "I don't need this right now" → "Totally fine. Is there anything about your current plan or network that isn't working perfectly for you? Even a small issue is worth solving."

## KNOWLEDGE BASE
When KNOWLEDGE BASE CONTEXT is provided below, always use it to answer plan, billing, network, and policy questions. Do not guess or invent any details not present in the context.\
"""

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

# BCP-47 → native script name. Injected into the LANGUAGE RULE so the reply
# script is pinned deterministically for EVERY language, not just the inline
# examples — mirrors the harness's per-language script map. Sarvam's bulbul TTS
# pronounces native script authentically, so wrong/romanized script is the main
# cause of bad pronunciation.
LANG_SCRIPTS: dict[str, str] = {
    "en-IN": "Latin",
    "hi-IN": "Devanagari",
    "mr-IN": "Devanagari",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "kn-IN": "Kannada",
    "bn-IN": "Bengali",
    "gu-IN": "Gujarati",
    "pa-IN": "Gurmukhi",
    "ml-IN": "Malayalam",
    "or-IN": "Odia",
}


# Localized "sorry, please repeat" lines. Spoken when a turn produces no visible
# content (e.g. a reasoning model spent its whole budget on hidden reasoning) so
# the caller hears a graceful prompt instead of dead air. Keyed by BCP-47.
_FALLBACK_PHRASES: dict[str, str] = {
    "en-IN": "Sorry, I didn't catch that. Could you say it again?",
    "hi-IN": "माफ़ कीजिए, मैं समझ नहीं पाया। क्या आप दोबारा बता सकते हैं?",
    "mr-IN": "माफ करा, मला समजलं नाही. तुम्ही पुन्हा सांगू शकता का?",
    "ta-IN": "மன்னிக்கவும், எனக்குப் புரியவில்லை. மீண்டும் சொல்ல முடியுமா?",
    "te-IN": "క్షమించండి, నాకు అర్థం కాలేదు. మళ్ళీ చెప్పగలరా?",
    "kn-IN": "ಕ್ಷಮಿಸಿ, ನನಗೆ ಅರ್ಥವಾಗಲಿಲ್ಲ. ಮತ್ತೊಮ್ಮೆ ಹೇಳಬಹುದೇ?",
    "bn-IN": "দুঃখিত, আমি বুঝতে পারিনি। আপনি কি আবার বলবেন?",
    "gu-IN": "માફ કરશો, મને સમજાયું નહીં. શું તમે ફરી કહી શકશો?",
    "pa-IN": "ਮਾਫ਼ ਕਰਨਾ, ਮੈਨੂੰ ਸਮਝ ਨਹੀਂ ਆਇਆ। ਕੀ ਤੁਸੀਂ ਦੁਬਾਰਾ ਕਹਿ ਸਕਦੇ ਹੋ?",
    "ml-IN": "ക്ഷമിക്കണം, എനിക്ക് മനസ്സിലായില്ല. ഒന്നുകൂടി പറയാമോ?",
    "or-IN": "କ୍ଷମା କରନ୍ତୁ, ମୁଁ ବୁଝିପାରିଲି ନାହିଁ। ଆପଣ ପୁଣି କହିପାରିବେ କି?",
}


def _fallback_phrase(language: str) -> str:
    """Localized 'sorry, please repeat' line, defaulting to English."""
    return _FALLBACK_PHRASES.get(language, _FALLBACK_PHRASES["en-IN"])


# Localized "we're at capacity, try again shortly" lines. Spoken when the voice
# LLM's free tier / rate limit is exhausted (a 429 that survives the bounded
# retry). This is a DISTINCT situation from "didn't catch that": the caller did
# nothing wrong, the service is momentarily unavailable — so we say so honestly
# instead of implying a transcription problem. Core demo languages only; any
# other detected language falls back to English (see _busy_phrase).
_BUSY_PHRASES: dict[str, str] = {
    "en-IN": "Sorry, we're experiencing very high demand right now. Please try calling again in a little while.",
    "hi-IN": "माफ़ कीजिए, अभी हमारे पास बहुत ज़्यादा कॉल्स आ रही हैं। कृपया थोड़ी देर बाद दोबारा कॉल करें।",
    "mr-IN": "माफ करा, सध्या आमच्याकडे खूप गर्दी आहे. कृपया थोड्या वेळाने पुन्हा कॉल करा.",
}


def _busy_phrase(language: str) -> str:
    """Localized 'service at capacity / limit reached' line, defaulting to English."""
    return _BUSY_PHRASES.get(language, _BUSY_PHRASES["en-IN"])


async def _empty_aiter():
    """An async iterator that yields nothing — used when the LLM stream could not
    be opened, so the streaming loop is a no-op and the empty-reply guard fires."""
    return
    yield  # pragma: no cover — makes this a generator, never reached


# ─────────────────────────────────────────────────────────────────────────────
# Explicit language-switch detection
# ─────────────────────────────────────────────────────────────────────────────
# WHY THIS EXISTS: STT auto-detects the language of the AUDIO every turn, and the
# LANGUAGE RULE forces the reply into that language. So a user asking "speak in
# Hindi" *in English* got an English reply forever — the request is content, but
# it's spoken in English, so detection wins. This detector reads the transcript,
# and if the user explicitly asks to switch, we LOCK the reply language until
# they ask again (persisted in AgentState.locked_language). Deterministic, no
# LLM, no tokens — per the token budget in CLAUDE.md §4.
#
# Recognise the target language by NAME in three forms: English ("Hindi"),
# romanized self-name ("hindi"), and native endonym ("हिंदी"). Keyed by BCP-47.
_LANG_ALIASES: dict[str, set[str]] = {
    "en-IN": {"english", "angrezi", "angreji", "इंग्लिश", "इंग्रजी",
              "अंग्रेजी", "अंग्रेज़ी", "इंग्रज़ी"},
    "hi-IN": {"hindi", "हिंदी", "हिन्दी"},
    "mr-IN": {"marathi", "मराठी"},
    "ta-IN": {"tamil", "தமிழ்"},
    "te-IN": {"telugu", "తెలుగు"},
    "kn-IN": {"kannada", "ಕನ್ನಡ"},
    "bn-IN": {"bengali", "bangla", "বাংলা"},
    "gu-IN": {"gujarati", "ગુજરાતી"},
    "pa-IN": {"punjabi", "panjabi", "ਪੰਜਾਬੀ"},
    "ml-IN": {"malayalam", "മലയാളം"},
    "or-IN": {"odia", "oriya", "ଓଡ଼ିଆ"},
}

# Switch-request VERBS/cues that must appear NEAR the language name for a switch
# to fire. Deliberately does NOT include a bare "in" — "in" alone is a common
# preposition ("available in Tamil Nadu") and matching it was the cause of a
# false-lock regression. English verbs, romanized Hindi/Marathi, and native
# Devanagari imperatives (बोल/कहो/बात कर).
# Only strong, unambiguous switch cues. Deliberately EXCLUDES weak verbs like
# "want"/"use"/"prefer": "I want a plan in Gujarati movies" names a language as
# an adjective, not a request — treating "want … <lang>" as a switch caused false
# locks. Speech verbs + "switch/change to" are the reliable signals; we favour
# precision, since a false lock derails the whole call while a missed switch just
# makes the user rephrase.
_SWITCH_VERB = (
    r"speak|talk|say|reply|respond|answer|converse|switch(?:\s+to)?"
    r"|change(?:\s+to)?"
    r"|bol(?:o|iye|na)?|baat\s+kar|mein\s+baat|bola|bolaycha"
    r"|बोल|कहो|कहिए|बात\s*कर|में\s*बोल|मध्ये\s*बोल"
)

# How many characters may sit between the switch verb and the language name for
# them to count as "adjacent" (one request, not two unrelated clauses).
_SWITCH_WINDOW = 30


def detect_language_switch(transcript: str) -> str | None:
    """
    Return a BCP-47 code (e.g. "hi-IN") if the transcript is an explicit request
    to switch the reply language, else None.

    A switch fires ONLY when a switch verb ("speak", "switch to", "bolo", native
    बोल …) appears ADJACENT to a language name — within ~{window} chars, in
    either order (English "speak in Hindi" vs. Hindi "हिंदी में बोलो"). Mere
    co-occurrence is not enough: "Is BharatConnect available in Tamil Nadu?"
    names a language but has no switch verb near it, so it must NOT lock the
    reply into Tamil. Deterministic, no LLM, no tokens (CLAUDE.md §4).
    """
    if not transcript:
        return None
    low = transcript.lower()

    for code, aliases in _LANG_ALIASES.items():
        for alias in aliases:
            # Latin aliases match case-insensitively with word boundaries; native
            # script matches raw (no word boundaries in Indic scripts).
            if alias.isascii():
                hay = low
                name = re.escape(alias.lower())
                name_pat = rf"\b{name}\b"
            else:
                hay = transcript
                name_pat = re.escape(alias)

            # Verb-then-name (e.g. "speak in Hindi") OR name-then-verb (e.g.
            # "हिंदी में बोलो"), within the adjacency window. [^.!?] stops the
            # match from spanning across sentence boundaries.
            w = _SWITCH_WINDOW
            verb_then_name = rf"(?:{_SWITCH_VERB})[^.!?]{{0,{w}}}?{name_pat}"
            name_then_verb = rf"{name_pat}[^.!?]{{0,{w}}}?(?:{_SWITCH_VERB})"
            if re.search(verb_then_name, hay, re.IGNORECASE) or \
               re.search(name_then_verb, hay, re.IGNORECASE):
                return code

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definitions  (Feature 9)
# ─────────────────────────────────────────────────────────────────────────────

# Tool schema passed to the Groq API
# Tools are DISABLED for the BharatConnect sales agent. The web_search tool
# (DuckDuckGo) was causing two serious problems:
#   1. The agent called it to look up BharatConnect plan/network facts — but
#      BharatConnect is a private/fictional brand, so search returned junk AND
#      the agent spoke a useless "Let me look that up for you" bridge phrase.
#   2. Each search is a slow external HTTP call (~9-11s), which was the cause of
#      the wildly inconsistent voice latency.
# All factual grounding comes from the RAG knowledge base instead. Leaving this
# as an empty list keeps the tool-call code paths intact but inert.
TOOLS: list = []


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


# Agent-initiated hangup: the LLM appends this token to its final goodbye when
# the customer has confirmed they have no further questions. It is a CONTROL
# signal — it must never be spoken aloud. We tolerate case and surrounding
# brackets/whitespace the model might add (e.g. "[END_CALL]", "[end call]").
_END_CALL_RE = re.compile(r'\[?\s*end[\s_-]*call\s*\]?', re.IGNORECASE)


def _has_end_call(text: str) -> bool:
    """True if the model signalled it wants to end the call."""
    return bool(_END_CALL_RE.search(text or ""))


def _strip_end_call(text: str) -> str:
    """Remove the [END_CALL] control token so it is never sent to TTS."""
    return _END_CALL_RE.sub("", text or "").strip()


# Primary split. Two boundary kinds:
#   1. Latin/full-width endings (. ? ! ？ ！) — require trailing whitespace/newline
#      so we don't split decimals ("299.00") or abbreviations ("Mr. Sharma").
#   2. Devanagari danda / double danda (। ॥) — HARD boundary, split even with NO
#      trailing space. Indic typography often abuts the danda to the next word
#      ("नमस्ते।आपकी…"); requiring whitespace missed those, delaying first audio
#      until the 40-word fallback (Bug #20). Dandas never appear inside numbers.
# Newlines also trigger a flush.
_SENTENCE_END_RE = re.compile(r'(?<=[.?!？！])\s+|(?<=[।॥])|\n+')

# Secondary split: commas/semicolons — only used when a clause is already
# long enough (≥4 words) to be a natural speech pause.  Sarvam TTS internally
# splits on commas and returns multiple `audios` entries; sending a pre-split
# chunk guarantees a single clean audio clip with no concatenation needed.
_COMMA_RE = re.compile(r'(?<=[,;،、])\s+')
_MIN_WORDS_BEFORE_COMMA_SPLIT = 3   # don't split "हाय, नमस्कार" but do split longer clauses


def _flush_sentences(buffer: str):
    """
    Extract speakable chunks from a streaming token buffer.

    Pass 1 — split on sentence-ending punctuation (. ? ! । ॥ ？ ！) or newlines.
    Pass 2 — for any chunk that still contains a comma/semicolon, further split
             on that comma *only if* the text before it has ≥4 words.  This
             prevents Sarvam TTS from receiving comma-heavy sentences that it
             would internally split into multiple `audios` entries (causing the
             WAV concatenation path and potential audio truncation).

    The last piece from the primary split is kept as the remainder
    (may be mid-sentence) and is never comma-split yet.

    Returns: (list_of_chunks, remaining_buffer)
    """
    parts = _SENTENCE_END_RE.split(buffer)
    if len(parts) <= 1:
        return [], buffer

    complete = [p.strip() for p in parts[:-1] if p.strip()]
    remaining = parts[-1].strip()

    result = []
    for chunk in complete:
        result.extend(_comma_split(chunk))
    return result, remaining


def _comma_split(text: str) -> list:
    """
    Split *text* on commas/semicolons, but only when the clause before the
    split point is ≥ _MIN_WORDS_BEFORE_COMMA_SPLIT words long.
    Always returns at least one element (the original text if no split applied).
    """
    sub_parts = _COMMA_RE.split(text)
    if len(sub_parts) <= 1:
        return [text]

    out     = []
    current = ""
    for part in sub_parts:
        candidate = (current + ", " + part).strip() if current else part
        # Count words (works for both Latin-script and Indic scripts separated by spaces)
        if current and len(current.split()) >= _MIN_WORDS_BEFORE_COMMA_SPLIT:
            out.append(current.strip())
            current = part
        else:
            current = candidate
    if current.strip():
        out.append(current.strip())
    return out if out else [text]


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
    resp = await _voice_llm.chat.completions.create(
        model=VOICE_LLM_MODEL,
        messages=[{
            "role": "user",
            "content": (
                "Summarize this conversation in 3-4 sentences, capturing the key "
                f"topics, decisions, and context:\n\n{history_text}"
            )
        }],
        **_voice_reasoning_kwargs(),
        max_tokens=300,
        temperature=0.3,
    )
    summary = resp.choices[0].message.content or ""
    logger.info(f"Summary generated: {summary[:80]!r}…")

    await agent_graph.aupdate_state(config, {"summary": summary}, as_node="llm")


# ─────────────────────────────────────────────────────────────────────────────
# State Save Helper
# ─────────────────────────────────────────────────────────────────────────────

async def _save_turn(config: dict, user_text: str, ai_text: str, new_turn_count: int,
                     locked_language: str = ""):
    """
    Persist the latest user+AI message pair to LangGraph MemorySaver and
    increment the turn counter. Triggers summarization every 20 turns.

    locked_language: the sticky reply-language lock (see detect_language_switch).
    Written every turn so an active lock survives across turns until the user
    asks to switch again; "" means no lock.

    as_node="llm" is required: LangGraph needs to know which node made the
    update so it can determine the next edge (llm → END in our graph).
    Without it, LangGraph raises "Ambiguous update, specify as_node".
    """
    await agent_graph.aupdate_state(
        config,
        {
            "messages":        [HumanMessage(content=user_text), AIMessage(content=ai_text)],
            "output":          ai_text,
            "turn_count":      new_turn_count,
            "locked_language": locked_language,
        },
        as_node="llm",
    )

    # Feature 8: summarize every 20 turns to keep context manageable.
    # Run it DETACHED (not awaited): summarization is a full extra non-streaming
    # Groq round-trip, and awaiting it here delayed _save_turn returning — which
    # gates LLMTurnDoneFrame and the TTS delivery loop's clean exit (Bug #17).
    # The summary is only consumed on a LATER turn, so it need not finish before
    # this turn completes. Errors are logged, never surfaced to the caller.
    if new_turn_count > 0 and new_turn_count % 20 == 0:
        async def _bg_summarize():
            try:
                await _summarize_history(config)
            except Exception as exc:
                logger.warning(f"_save_turn: background summarization failed (non-fatal): {exc}")
        asyncio.create_task(_bg_summarize(), name="dream-summarize")


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

    resp = await _voice_llm.chat.completions.create(
        model=VOICE_LLM_MODEL,
        messages=api_messages,
        **_voice_reasoning_kwargs(),
        temperature=0.7,
        max_tokens=200,
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

# ─────────────────────────────────────────────────────────────────────────────
# RAG gate — decide whether a query actually needs knowledge-base retrieval
# ─────────────────────────────────────────────────────────────────────────────
#
# Running RAG on EVERY turn is wasteful and dangerous on a tight TPM budget:
# injecting 5-7 full chunks into the prompt for a query like "yes" or "okay"
# adds thousands of useless tokens and can blow the per-minute token limit
# (the 413 "Request too large" error). This rule-based gate (~0ms, no LLM,
# no tokens) only triggers retrieval when the query looks like it needs facts
# from the knowledge base — plans, prices, billing, network, policy, etc.

# Knowledge-seeking keywords (English + Hindi/Marathi). If ANY appears, retrieve.
_RAG_TRIGGER_KEYWORDS: set[str] = {
    # plans / pricing / billing
    "plan", "plans", "price", "pricing", "cost", "rate", "recharge", "pack",
    "package", "packages",
    "data", "gb", "validity", "prepaid", "postpaid", "bill", "billing",
    "payment", "pay", "balance", "refund", "offer", "discount", "cheap",
    "expensive", "add-on", "addon", "topup", "top-up",
    # network / tech
    "network", "signal", "5g", "4g", "speed", "slow", "internet", "coverage",
    "connectivity", "fiber", "fibre", "broadband", "wifi", "wi-fi", "tower",
    "roaming", "outage", "down",
    # policy / support
    "otp", "fraud", "kyc", "complaint", "port", "mnp", "cancel", "sim",
    "policy", "refund", "terms", "verification",
    # competitive / switching
    "jio", "airtel", "vi", "vodafone", "idea", "bsnl", "switch", "compare",
    "comparison", "better", "competitor", "operator", "provider",
    # generic info-seeking
    "how much", "what is", "tell me about", "details", "feature", "benefit",
    # Hindi
    "प्लान", "रिचार्ज", "कीमत", "डेटा", "नेटवर्क", "बिल", "पैसे", "ओटीपी",
    "स्पीड", "इंटरनेट", "योजना", "बैलेंस",
    # Marathi
    "योजना", "रिचार्ज", "किंमत", "नेटवर्क", "वेग", "शिल्लक",
    # English words users often SPEAK but STT renders in Devanagari
    # (Hinglish). Without these, a KB question like "पैकेजेस के बारे में"
    # silently skips retrieval and the agent has no facts to answer with.
    "पैकेज", "पैकेजेस", "प्राइस", "प्राइसेस", "टॉप", "प्लान्स", "रेट",
    "डेटा", "रिचार्ज", "ऑफर", "डिस्काउंट", "प्रीपेड", "पोस्टपेड",
    "ब्रॉडबैंड", "फाइबर", "फ़ाइबर", "स्पीड", "प्लैन",
}

# Pure confirmations / chitchat — never need RAG even if short.
_RAG_SKIP_PHRASES: set[str] = {
    "yes", "no", "okay", "ok", "thanks", "thank you", "sure", "fine",
    "alright", "great", "hello", "hi", "hmm", "yeah", "yep", "nope",
    "हाँ", "नहीं", "ठीक है", "धन्यवाद", "हो", "बरोबर", "नाही",
}


def _should_retrieve(query: str) -> bool:
    """
    Rule-based RAG gate. Returns True only when the query plausibly needs
    knowledge-base facts. Zero latency, zero tokens, fully predictable.
    """
    q = query.strip().lower()
    if not q:
        return False
    # Skip pure confirmations / greetings-back
    if q in _RAG_SKIP_PHRASES:
        return False
    # Trigger if any knowledge keyword appears anywhere in the query
    return any(kw in q for kw in _RAG_TRIGGER_KEYWORDS)


@_traceable(name="rag_retrieval", tags=["retrieval"])
async def _retrieve_context(query: str, language: str) -> list:
    """
    Phase 4: Standalone @traceable wrapper around RetrievalPipeline.retrieve().

    Appears as a child span "rag_retrieval" in LangSmith under each
    stream_agent trace — shows the query, number of chunks returned, and
    top scores.  Returns an empty list when no pipeline is wired in.

    top_k reduced to 3 (from 5) to keep the injected context small enough to
    stay under the per-minute token limit on the free Groq tier.
    """
    if _retrieval_pipeline is None:
        return []
    return await _retrieval_pipeline.retrieve(query, language, top_k=3)


@_traceable(name="stream_agent", tags=["live_call"])
async def stream_agent(
    user_text: str,
    thread_id: str,
    emotion_hint: str = "neutral",
    language: str = "en-IN",
    _meta_out: dict | None = None,
) -> AsyncGenerator[str, None]:
    """
    Feature 3 (Streaming TTS): Async generator that yields sentence chunks
    from the LLM token stream in real-time — no waiting for a full sentence.

    Phase 2 addition: RAG context is retrieved before the LLM call and injected
    into the system prompt.  If _meta_out dict is provided, it is populated with:
        retrieved_docs : list[dict]   — chunks used for context (for trace recording)
        tool_calls     : list[dict]   — tool calls executed this turn

    Flow:
      1. Load conversation history from LangGraph MemorySaver.
      2. Build the API message list (with optional summary for Feature 8).
      2a. [Phase 2] RAG: retrieve knowledge-base chunks, inject into system prompt.
      3. Open a SINGLE streaming call with tool support (Feature 9).
      4. If tool call detected: yield bridging phrase, execute, stream follow-up.
      5. After all chunks are yielded, persist full response to LangGraph state.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # ── Greeting shortcut — bypass RAG/history for the opening line ───────────
    if user_text == "__greeting__":
        # Natural spelling — SarvamTTSService._normalize_pronunciation rewrites
        # "BharatConnect"→"भारत Connect" and "Suhas"→"सुहास" for correct Indic
        # pronunciation. Keep the phonetic mapping in ONE place (the TTS map),
        # not duplicated here.
        greeting = "Hello! Thank you for calling BharatConnect. This is Suhas. How can I help you today?"
        logger.info("stream_agent: emitting agent-initiated greeting")
        yield greeting
        # Persist the greeting into LangGraph memory as the first AI turn
        await agent_graph.aupdate_state(
            config,
            {"messages": [AIMessage(content=greeting)], "turn_count": 1, "summary": ""},
        )
        return

    # ── 1. Load history ───────────────────────────────────────────────────────
    snapshot = await agent_graph.aget_state(config)
    if snapshot and snapshot.values:
        existing_msgs: List[BaseMessage] = snapshot.values.get("messages", [])
        turn_count: int                  = snapshot.values.get("turn_count", 0)
        summary: str                     = snapshot.values.get("summary", "")
        locked_language: str             = snapshot.values.get("locked_language", "")
    else:
        existing_msgs   = []
        turn_count      = 0
        summary         = ""
        locked_language = ""

    # ── 1a. Explicit language switch ──────────────────────────────────────────
    # If the user asked to change languages ("speak in Hindi"), lock the reply
    # language until they ask again. The lock overrides STT auto-detection so a
    # request spoken in one language can switch the reply to another.
    requested = detect_language_switch(user_text)
    if requested:
        locked_language = requested
        logger.info(f"stream_agent: language locked → {requested} (explicit request)")

    # Effective reply language: an active lock wins over per-turn detection.
    effective_language = locked_language or language

    # ── 2. Build system prompt ────────────────────────────────────────────────
    system = BASE_SYSTEM_PROMPT + EMOTION_ADDENDA.get(emotion_hint, "")

    # Phase 6: append Dream Engine approved improvements (non-fatal if unavailable)
    # Cap at 3 addenda — every one is added to EVERY prompt on EVERY turn, so an
    # unbounded list silently bloats the token count and contributed to the 413
    # "request too large" error. 3 most-relevant improvements is plenty.
    try:
        addenda = (await _load_prompt_addenda())[:3]
        for addendum in addenda:
            system += f"\n{addendum}"
        if addenda:
            logger.debug(f"stream_agent: {len(addenda)} dream addenda applied")
    except Exception as _addenda_exc:
        logger.warning(f"stream_agent: addenda load failed (non-fatal): {_addenda_exc}")

    # Language instruction: tell the LLM which language to reply in.
    # effective_language = an explicit lock (if the user asked to switch) else
    # the STT-detected language. Reply ONLY in it — no mixing, no switching.
    lang_name = LANG_NAMES.get(effective_language, effective_language)
    # Pin the exact script per language (harness-style). Falls back to a generic
    # "native script of {lang}" instruction if the code isn't in the map.
    script_name = LANG_SCRIPTS.get(effective_language)
    script_clause = (
        f" using the {script_name} script only"
        if script_name else f" using the native script of {lang_name} only"
    )
    system += (
        f"\n\nLANGUAGE RULE (mandatory): Reply ENTIRELY in {lang_name}{script_clause}."
        f" Do NOT transliterate, romanize, or mix in any other language or script."
        f" Even if the user's input was in Roman/Latin script,"
        f" your reply must be in proper {lang_name} native script."
        f" Violating this rule is not allowed under any circumstances."
        # Brand name must stay literal ASCII so the TTS pronunciation map can
        # catch it. If written in native script (e.g. Malayalam), the map's
        # \\bBharatConnect\\b regex misses and TTS mangles the brand.
        f" EXCEPTION: always write the company name as the exact ASCII text"
        f" 'BharatConnect' — never transliterate it into another script."
    )

    if summary:
        system += f"\n\n[Earlier conversation summary]: {summary}"

    # ── Phase 2+4: RAG — parallel retrieval + LLM open ──────────────────────
    # Strategy: kick off RAG retrieval as a background task, then immediately
    # build the base api_messages and open the Groq streaming connection.
    # Retrieval (~150-300ms on a warm embedding cache) races the Groq TTFT
    # (~300-600ms).  In most cases RAG resolves before the first token arrives
    # so we can patch the context in before the LLM has said anything.
    # If RAG is slower, the base prompt (without context) answers and RAG
    # context is available for the next turn via the trace store.
    # This shaves 200-400ms off the perceived voice latency.

    retrieved_docs: list[dict] = []

    # Start RAG in background (non-blocking) — ONLY if the query actually needs
    # knowledge-base facts. Chitchat / confirmations skip RAG entirely, which
    # keeps the prompt small and avoids blowing the per-minute token limit.
    _rag_task = None
    if _retrieval_pipeline is not None and _should_retrieve(user_text):
        _rag_task = asyncio.create_task(_retrieve_context(user_text, effective_language))
    else:
        logger.debug(f"stream_agent: RAG skipped (no knowledge keywords in {user_text[:40]!r})")

    # Build base api_messages WITHOUT RAG context yet
    api_messages = [{"role": "system", "content": system}]

    visible_msgs = existing_msgs[-4:] if summary and len(existing_msgs) > 4 else existing_msgs
    for msg in visible_msgs:
        if isinstance(msg, HumanMessage):
            api_messages.append({"role": "user",      "content": msg.content})
        elif isinstance(msg, AIMessage):
            api_messages.append({"role": "assistant", "content": str(msg.content)})

    api_messages.append({"role": "user", "content": user_text})

    # Wait for RAG to finish before opening the Groq call.
    # We give it a tight timeout (1.5s) — on cache hit it's ~5ms,
    # on a cold embed it's ~300-500ms.  If it times out we proceed without context.
    if _rag_task is not None:
        try:
            chunks = await asyncio.wait_for(_rag_task, timeout=1.5)
            if chunks:
                # Truncate each chunk's content so the injected context stays
                # small. Full table_full/callout chunks can be 1-2k chars each;
                # 600 chars keeps the essential facts while protecting the TPM
                # budget. The LLM only needs the figures, not the whole table.
                context_block = "\n\n".join(
                    f"[{c.source} | {c.chunk_type}]\n{c.content[:600]}" for c in chunks
                )
                # Patch RAG context into the system message now (before Groq call)
                api_messages[0]["content"] += (
                    f"\n\nKNOWLEDGE BASE CONTEXT:\n{context_block}"
                    "\n\nBase your answer on the above context. "
                    "If the answer is not in the context, say so honestly."
                )
                retrieved_docs = [
                    {"doc_id": c.doc_id, "content": c.content[:200], "score": c.score}
                    for c in chunks
                ]
                logger.info(
                    f"stream_agent: RAG injected {len(chunks)} chunks "
                    f"(top score={chunks[0].score:.3f}, top_type={chunks[0].chunk_type})"
                )
            else:
                logger.debug("stream_agent: RAG returned no chunks (knowledge base may be empty)")
        except asyncio.TimeoutError:
            logger.warning("stream_agent: RAG timed out (>1.5s) — proceeding without context")
        except Exception as _rag_exc:
            logger.warning(f"stream_agent: RAG retrieval failed (non-fatal): {_rag_exc}")
    # ─────────────────────────────────────────────────────────────────────────

    logger.debug(
        f"stream_agent: {len(api_messages)} messages, emotion={emotion_hint!r}, "
        f"rag_chunks={len(retrieved_docs)}, summary={'yes' if summary else 'no'}"
    )

    # ── 3. Single streaming call — handles both content and tool-call paths ───
    # 413 safety net: if the request is still too large for the per-minute token
    # limit (TPM), drop the RAG context block and retry once. A slightly less
    # grounded answer beats dead silence on the call.
    async def _open_stream(messages):
        # Only pass tool params if tools are actually enabled — passing an empty
        # tools list can error on some API versions, and omitting them avoids any
        # accidental tool-calling overhead.
        tool_kwargs = {"tools": TOOLS, "tool_choice": "auto"} if TOOLS else {}
        return await _voice_llm.chat.completions.create(
            model=VOICE_LLM_MODEL,
            messages=messages,
            **tool_kwargs,
            # _voice_reasoning_kwargs() adds reasoning_effort="low" only for
            # gpt-oss models; Gemini (the default voice model) needs no reasoning
            # tax — it emits visible tokens immediately, which is why first-token
            # latency is low. max_tokens=200 leaves ample room for a short reply.
            **_voice_reasoning_kwargs(),
            max_tokens=200,
            temperature=0.5,
            stream=True,
        )

    stream = None
    # Set when the voice LLM's rate limit / free-tier quota is exhausted and the
    # bounded retry still fails. Drives a distinct "we're at capacity" message
    # (vs the generic "didn't catch that") in the empty-reply guard below.
    quota_exhausted = False
    try:
        stream = await _open_stream(api_messages)
    except Exception as _llm_exc:
        _msg = str(_llm_exc).lower()
        if "413" in _msg or "too large" in _msg:
            logger.warning(
                "stream_agent: request too large (TPM) — retrying without RAG context"
            )
            # Rebuild the system message without the injected knowledge block.
            stripped_system = system  # the pre-RAG system prompt
            api_messages[0]["content"] = stripped_system
            retrieved_docs = []
            try:
                stream = await _open_stream(api_messages)
            except Exception as _retry_exc:
                logger.error(f"stream_agent: 413 retry also failed: {_retry_exc}")
                stream = None
        elif any(k in _msg for k in ("429", "rate", "too many", "quota", "resource_exhausted", "exhausted")):
            # Voice LLM hit a rate/quota limit. Retry ONCE after a short bounded
            # sleep (the SDK already retried transient blips, so this is brief
            # real contention — e.g. a per-minute burst). If it STILL fails, the
            # free-tier quota is likely exhausted: flag it so the guard below
            # speaks a graceful "we're at capacity" message instead of dead air
            # or a misleading "didn't catch that". (Bug #13)
            logger.warning("stream_agent: rate/quota limit — one bounded retry after 1.5s")
            await asyncio.sleep(1.5)
            try:
                stream = await _open_stream(api_messages)
            except Exception as _retry_exc:
                logger.error(
                    f"stream_agent: rate/quota retry failed — voice LLM free tier "
                    f"likely exhausted: {_retry_exc}"
                )
                stream = None
                quota_exhausted = True
        else:
            raise

    word_buffer:    str  = ""
    full_response:  str  = ""
    is_tool_call:   bool = False
    # Accumulate tool-call JSON from streaming deltas (keyed by delta index)
    tool_calls_acc: dict = {}

    # stream is None only if opening it failed after retries (413/429) — leave
    # full_response empty so the empty-reply guard (§4b) speaks a localized
    # fallback instead of the caller hearing dead air.
    async for chunk in (stream or _empty_aiter()):
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

        # ── Content token: flush on sentence boundaries ───────────────────────
        elif delta.content:
            token         = delta.content
            full_response += token
            word_buffer   += token
            chunks, word_buffer = _flush_sentences(word_buffer)
            # Fallback: flush if the buffer grows very long with no sentence boundary.
            # Threshold is high (40 words) so it only fires for truly unpunctuated
            # responses — normal sentences with . ? ! । should split cleanly above.
            if not chunks and len(word_buffer.split()) >= 40:
                chunks    = [word_buffer.strip()]
                word_buffer = ""
            for c in chunks:
                clean = _strip_end_call(_strip_tool_markup(c))
                if clean.strip():
                    logger.info(f"stream_agent: yielding sentence → {clean.strip()!r}")
                    yield clean.strip()

    # Flush any remaining text after the stream ends (no-tool path)
    if not is_tool_call and word_buffer.strip():
        clean = _strip_end_call(_strip_tool_markup(_flush_all(word_buffer)))
        if clean:
            logger.info(f"stream_agent: yielding final fragment → {clean!r}")
            yield clean

    logger.info(f"stream_agent: done streaming. full_response={full_response!r}")

    # ── 4. Tool-call path: execute tool(s) then stream the follow-up ─────────
    if is_tool_call:
        tool_call_list = list(tool_calls_acc.values())
        logger.info(f"stream_agent: tool_calls detected — {[tc['function']['name'] for tc in tool_call_list]}")

        # Yield a bridging phrase immediately so TTS doesn't go silent.
        # Use a language-matched phrase so it doesn't break a non-English turn.
        SEARCH_PHRASES = {
            "hi-IN": "एक पल, मैं देखता हूँ।",
            "mr-IN": "एक मिनिट, मी शोधतो.",
            "ta-IN": "ஒரு நிமிடம், தேடுகிறேன்.",
            "te-IN": "ఒక్క నిమిషం, వెతుకుతున్నాను.",
            "kn-IN": "ಒಂದು ನಿಮಿಷ, ಹುಡುಕುತ್ತೇನೆ.",
            "bn-IN": "একটু অপেক্ষা করুন, খুঁজে দেখছি।",
            "gu-IN": "એક ક્ષણ, હું શોધું છું.",
            "pa-IN": "ਇੱਕ ਮਿੰਟ, ਮੈਂ ਲੱਭਦਾ ਹਾਂ।",
            "ml-IN": "ഒരു നിമിഷം, ഞാൻ നോക്കുന്നു.",
        }
        yield SEARCH_PHRASES.get(effective_language, "Let me look that up for you.")

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

        stream2 = await _voice_llm.chat.completions.create(
            model=VOICE_LLM_MODEL,
            messages=follow_up_messages,
            **_voice_reasoning_kwargs(),
            max_tokens=250,
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
            chunks, word_buffer2 = _flush_sentences(word_buffer2)
            for c in chunks:
                clean = _strip_end_call(_strip_tool_markup(c))
                if clean.strip():
                    yield clean.strip()

        if word_buffer2.strip():
            clean = _strip_end_call(_strip_tool_markup(_flush_all(word_buffer2)))
            if clean:
                yield clean

        full_response = SEARCH_PHRASES.get(effective_language, "Let me look that up for you.") + " " + full_response2

    # ── 4b. Empty-reply guard (reasoning starvation / model returned nothing) ─
    # A gpt-oss reasoning model can spend its entire max_tokens on hidden
    # chain-of-thought and emit no visible content. Then the loops above yielded
    # nothing → the caller hears dead air, and the `if clean_response` guard below
    # would skip _save_turn, losing this turn from history and corrupting later
    # context. Speak a localized "sorry, please repeat" instead, and make sure the
    # turn is persisted (full_response is now non-empty, so §6 saves it).
    if not _strip_end_call(full_response).strip():
        if quota_exhausted:
            # Free tier / rate limit hit and the retry failed — the service is
            # momentarily unavailable, not a transcription miss. Say so honestly.
            phrase = _busy_phrase(effective_language)
            logger.warning(
                "stream_agent: voice LLM free tier/quota exhausted — "
                f"emitting 'at capacity' message in {effective_language}"
            )
        else:
            phrase = _fallback_phrase(effective_language)
            logger.warning(
                "stream_agent: empty model reply (likely reasoning starvation) — "
                f"emitting fallback in {effective_language}"
            )
        full_response = phrase
        yield phrase

    # ── 5. Expose metadata for trace recording (Phase 3) ─────────────────────
    # Agent-initiated hangup: if the model appended [END_CALL] anywhere in its
    # reply, signal the pipeline to close the call after the goodbye audio plays.
    end_call = _has_end_call(full_response)
    if end_call:
        logger.info("stream_agent: [END_CALL] detected — agent will end the call after goodbye")
    if _meta_out is not None:
        _meta_out["end_call"] = end_call
        _meta_out["retrieved_docs"] = retrieved_docs
        _meta_out["tool_calls"] = [
            {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
            for tc in (tool_calls_acc.values() if is_tool_call else [])
        ]

    # ── 6. Save state ─────────────────────────────────────────────────────────
    # Strip the control token before persisting so it never re-enters the prompt
    # as conversation history on a later turn.
    clean_response = _strip_end_call(full_response).strip()
    if clean_response:
        try:
            await _save_turn(config, user_text, clean_response, turn_count + 1,
                             locked_language=locked_language)
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
