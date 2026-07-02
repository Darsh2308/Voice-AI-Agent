---
title: DreamSupport
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: other
short_description: The Self-Improving Customer Support AI Voice Bot
---

# DreamSupport — A Self-Improving Multilingual Voice AI Agent

DreamSupport is a real-time, full-duplex **voice sales agent** for *BharatConnect*, a
(fictional) Indian telecom. You call it in your browser and have a natural spoken
conversation — in **English, Hindi, Marathi, and 8 other Indian languages** — and it
listens, understands, retrieves grounded facts, replies in your language, and gently
tries to move you toward becoming a customer.

What makes it different from an ordinary voice bot: **it improves itself.** While no
one is on a call, an offline **Dream Engine** replays the day's conversations, grades
them, finds what went wrong, writes concrete prompt improvements, and feeds them back
into the live agent — with no restart and no human in the loop.

> **The agent talks to you.** The **Dream Engine makes it better at talking to you.**

---

## Table of Contents

1. [What DreamSupport Is](#1-what-dreamsupport-is)
2. [The Complete Voice Pipeline](#2-the-complete-voice-pipeline)
3. [Models Used — TTS, STT, LLM, Embeddings, RAG](#3-models-used)
4. [The RAG Knowledge System](#4-the-rag-knowledge-system)
5. [The Dream Engine — How Self-Improvement Works](#5-the-dream-engine)
6. [The Self-Improvement Feedback Loop (end to end)](#6-the-self-improvement-feedback-loop)
7. [Every Edge Case We Handle](#7-every-edge-case-we-handle)
8. [Architecture Map & Data Stores](#8-architecture-map--data-stores)
9. [Configuration Reference](#9-configuration-reference)
10. [Setup, Running & Deployment](#10-setup-running--deployment)
11. [API Reference](#11-api-reference)
12. [Project Structure](#12-project-structure)
13. [Tech Stack](#13-tech-stack)

---

## 1. What DreamSupport Is

A **multilingual, full-duplex voice sales agent** with three parts working together:

| Part | What it does |
|---|---|
| **Live voice pipeline** | Real-time browser voice call: mic → VAD → speech-to-text → LLM (with RAG) → text-to-speech → audio back, full-duplex with barge-in. |
| **RAG knowledge layer** | Grounds every factual answer in a curated BharatConnect knowledge base stored in a Qdrant vector database, so the agent never invents plan prices or policies. |
| **Dream Engine** | An offline self-improvement loop that runs only when the system is idle. It grades past conversations, discovers knowledge gaps, and writes prompt improvements the live agent picks up automatically. |

**The agent's persona:** "Suhas", a warm, concise BharatConnect sales rep. It answers the
customer's actual question first (from the knowledge base), then guides them one step
toward converting — without being pushy and without inventing facts.

**Design philosophy (production, not demo):**
- Every turn must feel instant — a phone call has a hard **~1.5–2s latency budget**.
- The voice agent (Gemini) and the Dream Engine (Groq) run on **separate LLM providers
  with independent free-tier pools**, so dreaming can never starve live calls. A hard
  daily token cap on dreaming remains as defense-in-depth.
- Every unhappy path (timeout, 429, 413, empty reply, disconnect, barge-in) is handled
  explicitly and degrades gracefully — the agent never goes silent on the customer.

---

## 2. The Complete Voice Pipeline

The whole system is built on **Pipecat**, a frame-based pipeline framework. Every piece
of data (an audio chunk, a transcript, a sentence, synthesized audio) is a typed
**Frame**. Each stage is a `FrameProcessor` that receives frames, does one job, and
pushes new frames downstream. This makes every component independently replaceable.

```
BROWSER (index.html)
  │  16-bit PCM mic audio  ─┐        ┌─  WAV audio + transcripts + status JSON
  ▼                        │  WebSocket /ws (full-duplex)         ▲
FastAPI server (app/main.py)  ── echo cooldown · barge-in · timeout · hangup ──┘
  │
  ▼   AudioRawFrame (raw PCM)
┌──────────────────────────────────────────────────────────────────────┐
│  Pipecat Pipeline  (app/pipecat_pipeline.py — one per connection)      │
│                                                                        │
│  VADProcessor  ──► SarvamSTTService ──► GroqLangGraphProcessor ──►      │
│  (Silero VAD)      (Sarvam ASR)          (LLM brain + memory)           │
│                                              │                          │
│                                              ▼                          │
│                                    stream_agent()  (langgraph_flow.py)  │
│                                     ├─ RAG gate → RetrievalPipeline      │
│                                     │             → Qdrant vector search │
│                                     └─ Gemini LLM (streaming, sentences) │
│                                              │                          │
│  ──► SarvamTTSService ──► OutputSink ────────┘                          │
│      (Sarvam TTS)         (asyncio.Queue → send loop)                   │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.1 Browser — audio capture

The browser (`index.html`) uses the Web Audio API:
- `getUserMedia({ echoCancellation: true })` opens the mic (echo cancellation is why the
  server-side echo cooldown can be as short as 0.3s).
- Captures ~4096-sample chunks (~85 ms) at the native rate (usually 48 kHz), converts
  float32 → 16-bit PCM, and streams them as **binary WebSocket frames**.
- Sends `{"type":"init","sampleRate":48000}` on connect and `{"type":"interrupt"}` on
  manual interrupt.

### 2.2 VADProcessor — Voice Activity Detection (Silero VAD)

Decides *when the user starts and stops speaking* using the **Silero VAD** neural network
(loaded once at module import, run on CPU). Algorithm:

1. Resample each browser chunk from 48 kHz → **16 kHz** (Silero + Sarvam both need 16 kHz).
2. Split into 512-sample windows, run each through Silero → speech probability (0.0–1.0).
   *Inference is offloaded to a thread pool so it never blocks the event loop (Bug #10).*
3. Buffer audio while probability > `SPEECH_THRESHOLD (0.5)`.
4. **Speech start** after `MIN_SPEECH_CHUNKS (3)` consecutive speech chunks (~0.25 s).
5. **Speech end** after `SILENCE_CHUNKS_NEEDED (12)` silent chunks (~1.0 s of quiet).
   *Tuned up from 0.42 s because it was cutting people off mid-thought.*
6. Hard cap at `MAX_BUFFER_CHUNKS (180)` (~15 s) to prevent infinite buffering.
7. Emit a `SpeechEndFrame` with the complete utterance as a 16 kHz WAV, then reset Silero's
   hidden state.

**Barge-in mode** (active while the AI is speaking): fires a `BargeInDetectedFrame` after
only `MIN_SPEECH_CHUNKS_BARGEIN (2)` chunks (~170 ms) so interruption feels instant, and
ends the utterance faster (`SILENCE_CHUNKS_BARGEIN (4)`).

**Emotion hint:** tracks per-utterance RMS energy; if energy > 2× the rolling baseline, it
emits `EmotionHintFrame("agitated")`.

### 2.3 SarvamSTTService — Speech to Text

- Uploads the utterance WAV to **Sarvam ASR** (`POST https://api.sarvam.ai/speech-to-text`).
- **Model `saaras:v3`** with `mode="transcribe"` (`saarika:v2.5` is legacy/deprecated),
  `language_code="unknown"` (always auto-detect, so mid-call language switches like
  English → Hindi → Marathi work correctly). Model + mode are config-driven
  (`SARVAM_STT_MODEL`, `SARVAM_STT_MODE`) — `mode` applies only to `saaras:*` models.
- **Language detection + romanized-language correction:** reads Sarvam's `language_code`,
  normalizes it to BCP-47 (`hi` → `hi-IN`, etc.). Because Sarvam often mislabels *romanized*
  Indian speech (Hinglish, romanized Marathi) as English, a **function-word marker scan**
  overrides `en-IN` when it finds ≥2 known markers of another language (word lists for Hindi,
  Marathi, Tamil, Telugu, Kannada, Punjabi, Gujarati, Bengali). Emits `LanguageDetectedFrame`.
- **Emotion hint:** if Sarvam returns confidence < 0.6, emits `EmotionHintFrame("hesitant")`.
- **Noise filter:** drops empty, ≤2-char, or filler-word transcripts ("yes", "ok", "hmm"…).
- Emits the user's text as `TranscriptDisplayFrame` (shown first, so the user's bubble
  appears before the AI audio) then `TranscriptionFrame` (triggers the LLM).

### 2.4 GroqLangGraphProcessor + `stream_agent()` — the brain

This is where the agent thinks, remembers, and grounds itself. On each user turn:

1. Emit `AIThinkingFrame(True)` → browser shows a typing indicator.
2. Call `stream_agent()` (in `app/langgraph_flow.py`), an **async generator that yields
   complete sentences** as the LLM streams them.
3. First sentence → emit `AIThinkingFrame(False)`, push each sentence as a `TextFrame` to
   TTS immediately (so audio starts before the LLM finishes).
4. On turn end: emit the full reply as a `TranscriptDisplayFrame`, **record a TurnTrace**
   (bounded to 1s so a slow DB never stalls the call — Bug #14), push `LLMTurnDoneFrame`,
   and — if the LLM signalled `[END_CALL]` — push `EndCallFrame`.

Inside `stream_agent()`:
- **System prompt** = base persona + emotion addendum + language instruction + up to **3
  approved Dream-Engine improvements** + earlier-conversation summary.
- **RAG gate** (`_should_retrieve`): rule-based, zero-latency, zero-token. Skips retrieval
  for pure confirmations/greetings ("yes", "no", "thanks", "हाँ", "नाही"…); triggers it only
  when knowledge keywords appear. RAG runs **in parallel** with a **1.5 s timeout** — if it's
  slow, the agent answers without it rather than making the customer wait.
- **Streaming + sentence splitting:** splits on `.?!` + whitespace, Devanagari danda `।॥`, or
  newlines; secondary split on commas only if the clause has ≥3 words; force-flushes at 40
  words with no boundary.
- **Greeting shortcut:** the sentinel `"__greeting__"` yields a fixed opening line so the
  agent greets the caller first without waiting for them to speak.

### 2.5 SarvamTTSService — Text to Speech

- Synthesizes each sentence via **Sarvam TTS** (`POST https://api.sarvam.ai/text-to-speech`).
- **Model `bulbul:v3`**, speaker **`simran`** (config-driven via `SARVAM_TTS_MODEL` /
  `SARVAM_TTS_SPEAKER`; note `bulbul:v3` uses its own speaker set — v2 names like
  `anushka` are rejected), `target_language_code` auto-switched by
  `LanguageDetectedFrame` so the AI replies in the user's language.
- **Pipeline-parallel design:** each sentence fires a **background API task immediately**;
  a separate in-order delivery loop pushes audio in arrival order as each resolves. This
  removed the bottleneck where LLM streaming was serialized behind TTS round-trips.
- **Pre-TTS text processing** (spoken audio only — transcript/logs keep original spelling):
  - **`num_to_words`** spells every digit into words *in the reply's language* ("two ninety
    nine rupees"), because Sarvam drops/mis-speaks bare numerals — critical for prices.
  - **Pronunciation map** rewrites brand/place names into Devanagari so Sarvam reads them
    natively ("BharatConnect" → "भारत कनेक्ट", "Suhas" → "सुहास", cities, competitors), and
    spells acronyms ("BSNL" → "B S N L", "VoLTE" → "V O L T E").
  - **Truncation** at sentence boundaries to respect Sarvam's 450-char limit.
- Emits `AIStatusFrame(True/False)` around speech and one `AIAudioFrame` (WAV) per sentence.

### 2.6 OutputSink & send loop

`OutputSink` puts browser-relevant frames onto an `asyncio.Queue`; the WebSocket `send_loop`
drains it and sends transcripts (text), audio (binary WAV), and status/thinking/language/
barge-in/hangup messages (JSON) to the browser.

---

## 3. Models Used

Every model in the stack, end to end:

| Role | Model | Provider | Notes |
|---|---|---|---|
| **Voice Activity Detection** | Silero VAD (`silero_vad`) | torch.hub (local, CPU) | Neural speech detection + barge-in. Pre-downloaded at Docker build. |
| **Speech-to-Text (ASR)** | `saaras:v3` (`mode=transcribe`) | Sarvam AI | `SARVAM_STT_MODEL`/`SARVAM_STT_MODE`. Auto-detects language every turn; 11 Indian languages. (`saarika:v2.5` is legacy.) |
| **LLM (live voice agent)** | `gemini-3.1-flash-lite` | Google (Gemini, OpenAI-compatible endpoint) | `VOICE_LLM_MODEL`. Fast first token (no reasoning tax), strong Indian-language + native-script, streaming + tools. |
| **LLM (Dream Engine)** | `openai/gpt-oss-120b` | Groq | `DREAM_LLM_MODEL`. A reasoning model — an asset for offline evaluation/self-critique where latency doesn't matter. |
| **Text-to-Speech (TTS)** | `bulbul:v3` (speaker `simran`) | Sarvam AI | `SARVAM_TTS_MODEL`/`SARVAM_TTS_SPEAKER`. Natural multilingual TTS; auto language switch. |
| **Embeddings (RAG)** | `intfloat/multilingual-e5-small` | sentence-transformers (local, CPU) | **384-dim**, multilingual. Free, runs on CPU. Pre-downloaded at build. |
| **Reranker (optional, off)** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | sentence-transformers | Enable with `RERANKER=true`. |

> **Model migration:** all model IDs live in `app/config.py` (`VOICE_LLM_MODEL`,
> `DREAM_LLM_MODEL`, `SARVAM_STT_MODEL`, `SARVAM_TTS_MODEL`, `SARVAM_TTS_SPEAKER`). Change
> them via config/`.env` only — never hardcode IDs in app code.

**Why separate providers for voice and dream?** The two workloads have opposite needs.
The **voice** path is latency-critical and short-form, so it uses **Gemini Flash-Lite** — it
emits visible tokens immediately (no hidden-reasoning delay) and handles Indian languages /
native script well. The **Dream Engine** is offline and reasoning-heavy (grading turns,
judging prompt proposals), so it uses Groq's **`gpt-oss-120b`**, a reasoning model where
latency is irrelevant and depth is an asset. Running them on **different providers** also
makes their free-tier token pools independent — dreaming can never exhaust the live voice
quota. `reasoning_effort="low"` is passed **only** for gpt-oss models (Gemini's
OpenAI-compatible endpoint rejects the param); it re-appears automatically if
`VOICE_LLM_MODEL` is ever pointed back at a gpt-oss model.

### LLM call parameters

| Call | Provider | Temperature | max_tokens | reasoning_effort | Streaming |
|---|---|---|---|---|---|
| Voice — first call | Gemini | 0.5 | 200 | n/a¹ | yes |
| Voice — follow-up after tool | Gemini | 0.7 | 250 | n/a¹ | yes |
| Conversation summarization | Gemini | 0.3 | 300 | n/a¹ | no |
| Dream Engine (JSON eval) | Groq | 0.3 | 512 (default) | low | no |
| Dream Engine (text) | Groq | 0.4 | 256 (default) | low | no |

¹ `reasoning_effort` is a gpt-oss param; it's omitted for Gemini and auto-included only if the voice model is switched back to a gpt-oss model.

---

## 4. The RAG Knowledge System

The agent must never invent a plan price or policy, so factual answers are grounded in a
curated knowledge base stored in **Qdrant Cloud**.

### 4.1 Embeddings
- Model **`intfloat/multilingual-e5-small`**, **384-dim**, multilingual, local/CPU (free).
- **E5 prefix convention** (required for this model): documents embedded as
  `"passage: <text>"`, queries as `"query: <text>"`.
- In-process **LRU cache** (max 512 entries) avoids re-embedding repeated text.
- Provider is swappable: `EMBEDDING_PROVIDER=openai` uses `text-embedding-3-small` instead.

### 4.2 Ingestion (`app/knowledge/ingestor.py`)
Turns 5 BharatConnect PDFs into retrievable chunks using **layout-aware `pdfplumber` parsing**:

| PDF | doc_id | topic |
|---|---|---|
| Company Overview | `KB-CORP-001` | policy |
| Policies and Terms | `KB-POL-002` | policy |
| Billing / Recharges / Plans | `KB-BILL-003` | billing |
| Network and Technology | `KB-NET-004` | network |
| Competitive Landscape | `KB-COMP-005` | competitive |

Chunking is **structure-aware**, not blind fixed-size:
- **Prose** split at headings; sub-split at ~1600 chars with ~240-char overlap; breadcrumb
  prefix (`BharatConnect > <doc> > <section>: `). `chunk_type="prose"`.
- **Catalogue tables** → one chunk per row, serialized as a self-contained sentence, with
  finer topics `plans_prepaid` / `plans_postpaid`. `chunk_type="table_row"`.
- **Comparison/matrix tables** → one atomic chunk. `chunk_type="table_full"`.
- **Critical callouts** (policy boxes: "never", "always", "agents must") → stored **twice**
  with different breadcrumbs for retrieval redundancy. `chunk_type="callout"`, `priority="critical"`.

Each chunk gets a deterministic UUID (from SHA-256 of its content) so re-ingesting is
idempotent. Upserted in batches of 32. Run it manually:

```bash
python -m app.knowledge.ingestor            # ingest all 5 PDFs
python -m app.knowledge.ingestor --file …   # a single file
python -m app.knowledge.ingestor --verify   # show chunk counts per doc_id
```

### 4.3 Retrieval (`app/knowledge/retriever.py`)
`RetrievalPipeline.retrieve(query, language, top_k)` runs these stages:

1. **Critical-chunk injection (deterministic, no vector search):** if the query hits purchase
   triggers ("recharge", "buy", "pay", "website"…) it force-injects the corporate purchase
   doc; if it hits OTP/fraud triggers ("otp", "cvv", "card number"…) it injects the security
   policy doc — score forced to 1.0 so safety-critical rules always surface.
2. **Topic pre-filter:** a rule-based keyword classifier tags the query as
   `billing` / `network` / `policy` / `competitive` and filters Qdrant to that topic (billing
   widens to include prepaid/postpaid plan rows). Ambiguous queries → no filter.
3. **Embed query** (`"query: …"`) and **vector search** the `knowledge_base` collection
   (Cosine distance).
4. **Low-confidence retry (Bug #9):** if the filtered search returns nothing or a weak top
   score, run one *unfiltered* search and merge — so an over-aggressive topic filter can't
   hide the right answer.
5. **Score filter** (`RAG_MIN_SCORE=0.25`), dedup vs critical chunks, optional cross-encoder
   rerank (off by default), and return `critical + similarity` chunks.

Retrieved chunks are truncated to **600 chars each** before injection into the prompt (token
budget), and the whole retrieval is subject to the **1.5 s parallel timeout** in `stream_agent`.

---

## 5. The Dream Engine

**The self-improvement brain.** A permanent background `asyncio.Task` that runs **only when
no customer is on a call**. When idle, it replays past conversations, grades them, finds
lessons, and writes improvements the live agent picks up automatically.

> Analogy: during the day the agent talks to customers; at night, while "asleep," it replays
> the day, learns from its mistakes, and wakes up a little better — no code change, no restart.

### 5.1 When it runs — the state machine (`app/dream/engine.py`)

```
PAUSED  ──(no customers for DREAM_IDLE_THRESHOLD_SECS = 300s)──►  DREAMING
   ▲                                                                  │
   └──────────────── customer connects (instant pause) ◄─────────────┘
```

- A counter `_active_sessions` is bumped by `customer_connected()` / `customer_disconnected()`
  (called from the WebSocket handler). Pausing uses an `asyncio.Event`.
- On the last disconnect it waits **300 s** before dreaming (via a cancellable timer) so brief
  reconnects / page reloads don't wake it. It re-checks the counter is still zero before starting.
- Every cycle checks the pause signal **after each unit of work**; on a customer connect it
  **saves a Qdrant checkpoint and returns instantly**, resuming from that bookmark next idle
  window. This is what keeps pausing invisible to the customer.
- The loop rotates the 5 cycles (1→2→3→4→5→…) with `DREAM_CYCLE_INTERVAL_SECS (300s)` between them.

### 5.2 The five cycles (`app/dream/cycles.py`)

An assembly line — each stage feeds the next:

**Cycle 1 — FailureAnalysis** — *grade the homework.*
Pulls unscored turns from `execution_traces` in batches of 10 and asks an LLM **sales-quality
judge** to score each turn 1–10 on three axes: **correctness**, **helpfulness**, and
**sales_progress** (did the turn *advance the sale*, or just answer and stall with "anything
else?"). Writes the score + issues back and flips `dream_processed=True`. Turns < 6 are flagged.

**Cycle 2 — RetrievalQualityAnalysis** — *did we have the right knowledge?*
For low-scoring turns, judges whether the RAG context was relevant. When the KB had no good
answer, logs a **knowledge gap** (and a better-query hint) to `improvement_log`. Marks turns so
they aren't re-analyzed (Bug #8).

**Cycle 3 — PromptImprovement** — *write a better instruction (the real magic).*
Clusters failing turns → asks the LLM to **propose one system-prompt addendum** per cluster →
a separate **LLM judge** checks it against held-out failures. Only proposals clearing
`MIN_JUDGE_SCORE (0.6)` are stored in `improvement_log` with `category="prompt", approved=True`.
This propose→judge→approve gate stops the agent teaching itself bad habits.

**Cycle 4 — SyntheticQueryGen** — *invent practice problems.*
From real failures, generates 3 test variants each (adversarial, edge-case, happy-path), stored
as `customer_id="synthetic"` traces — excluded from real metrics but available to Cycle 1 as a
self-built regression set. Capped at 30 source turns/run.

**Cycle 5 — MemoryConsolidation** — *tidy up.*
Marks profiles unseen for 90 days as stale, regenerates summaries for frequent callers (≥5
sessions), flags KB documents never retrieved, and cleans up old synthetic turns.

### 5.3 The safety guard — daily token budget (`app/dream/budget.py`)

The Dream Engine runs on Groq's `gpt-oss-120b`. Since the voice agent now runs on a
**separate provider** (Gemini), dreaming no longer shares a token budget with live calls —
so it gets the **full gpt-oss-120b free-tier day** (~200K tokens). `DreamTokenBudget`
remains as a safety ceiling (Groq's own 429 is the real backstop):
- **Before every LLM call:** `can_afford()` checks a conservative estimate against the cap;
  over-cap raises `_BudgetExhausted` and the whole run stops.
- **After every call:** records the *actual* Groq-reported tokens and **persists the running
  total to Qdrant**, so a mid-day restart doesn't reset the cap.
- **Resets** automatically at the UTC day rollover.
- Cap: **`DREAM_DAILY_TOKEN_BUDGET = 200000`** (full free-tier day). Lower it to make dreaming do less per day.

`_BudgetExhausted` and `_RateLimitHit` inherit from `BaseException` on purpose, so the cycles'
many `except Exception` blocks can't swallow them — they bubble straight to the engine, which
backs off **1 hour** (not the usual 5 min) before retrying, since the limit only resets daily.

---

## 6. The Self-Improvement Feedback Loop

The piece that makes this "self"-improvement rather than just analytics: **the live agent reads
the Dream Engine's approved improvements on every single turn.**

```
LIVE VOICE PATH (real time)
  User speaks → STT → stream_agent() [LLM + RAG] → TTS → User
                              │
                              ▼  (fire-and-forget, 1s timeout — never blocks the call)
                   record_turn(TurnTrace) ──► execution_traces  (dream_processed=False)
                              │
                              ▼  (offline, only when idle)
DREAM ENGINE
  Cycle 1 get_unprocessed_turns() → LLM grades → update_eval_score() (dream_processed=True)
  Cycle 2/3 → propose + judge → improvement_log (category="prompt", approved=True)
                              │
                              ▼  (next turn onward — no restart)
BACK INTO LIVE PATH
  stream_agent() loads up to 3 approved prompt addenda from improvement_log
  → appends them to the system prompt → the next customer talks to a better agent
```

**A `TurnTrace`** (written by `ExecutionTraceStore.record_turn`, in the `execution_traces`
collection with a dummy vector) captures everything the Dream Engine needs: `session_id`,
`turn_index`, `user_input`, `ai_response`, `detected_language`, `retrieved_docs`, `tool_calls`,
`latency_ms`, `emotion_hint`, `created_at`, plus the offline-filled `eval_score`, `eval_issues`,
`eval_dimensions`, and the `dream_processed` idempotency flag.

Live-path browsing of `/dream/knowledge-gaps` and `/dream/improvements` (see [API](#11-api-reference))
lets an operator watch what the engine has learned.

---

## 7. Every Edge Case We Handle

This system's CLAUDE.md records real hours lost to subtle failures. Here's everything guarded:

### Voice / streaming edge cases
| Edge case | Guard |
|---|---|
| **Echo (AI hears itself)** | `POST_AI_COOLDOWN_SECS = 0.3` — ignore mic audio for 0.3s after AI stops (browser WebRTC echo-cancel handles the rest). |
| **Barge-in (user talks over AI)** | Audio always flows even during playback; VAD fires `BargeInDetectedFrame` after ~170 ms → `interrupt()` cancels the current turn only (non-destructive), stops browser audio. |
| **Interrupt must not kill the pipeline** | `interrupt()` calls `cancel_turn()` (cancels in-flight TTS tasks + delivery loop) **not** `PipelineTask.cancel()` — the latter permanently ends the runner and made the agent go deaf after the first barge-in. |
| **Silent after barge-in** | `cancel_turn()` resets TTS turn state to pristine (`_llm_done=None`) so the next turn is detected as fresh and its audio is delivered. |
| **Mid-sentence "thinking" pause** | End-of-utterance silence window widened to ~1.0s (12 chunks) so natural pauses don't cut the customer off. |
| **VAD blocking the event loop** | Silero inference offloaded to a thread pool (Bug #10) so audio delivery/barge-in stay smooth. |
| **TTS round-trip bottleneck** | Sentences synthesized as concurrent background tasks with in-order delivery. |
| **Total TTS failure for a turn** | Delivery loop still emits `AIStatusFrame(False)` so the client isn't stuck on "thinking" (Bug #16). |
| **Sarvam splitting audio on commas** | Multiple returned clips concatenated into one WAV. |
| **Agent-initiated hangup cut off** | `[END_CALL]` → `EndCallFrame` forwarded **only after** goodbye audio is delivered; close grace = `min(12s, playback + 1s)`. |
| **Idle connection** | `timeout_watch` closes the socket after `INACTIVITY_TIMEOUT_SECS = 600` (10 min). |
| **Disconnect cleanup** | On any loop exit: end session, `dream_engine.customer_disconnected()`, stop pipeline, and delete this connection's LangGraph memory (Bug #7 — unbounded RAM growth). |
| **Numbers dropped by TTS** | `num_to_words` spells digits into words in the reply language before synthesis. |
| **Brand/name mispronunciation** | Devanagari pronunciation map (spoken-only). |

### LLM / token-budget edge cases
| Edge case | Guard |
|---|---|
| **413 "request too large" (prompt > TPM)** | Retry once with the RAG context stripped — smaller prompt, not a bigger model. |
| **429 / quota exhausted (live voice)** | Sleep 1.5s, retry once; if it still fails, the free tier is likely exhausted → the agent **speaks a localized "we're at capacity, try again shortly" message** (distinct from "didn't catch that"), never dead air. |
| **Empty reply (model returned nothing)** | Detected and replaced with a **localized fallback phrase** in the user's language (11 languages). |
| **Reasoning overhead (Dream / Groq only)** | `reasoning_effort="low"` on gpt-oss calls; omitted for Gemini (voice), which has no hidden-reasoning phase. |
| **Prompt bloat from Dream addenda** | Approved addenda capped at **3 per prompt** (an unbounded list caused a 413). |
| **RAG slow on the hot path** | RAG runs in parallel with a **1.5s timeout**; the turn proceeds without it if it's late. |
| **RAG wasting tokens on chitchat** | `_should_retrieve` gate skips retrieval for confirmations/greetings. |
| **Long conversations blowing context** | Summarize every **20 turns**; thereafter send only the summary + last 4 messages. |
| **Slow trace DB stalling a call** | `record_turn` bounded by a **1s timeout** (Bug #14); on timeout the trace is skipped, not awaited. |

### Dream Engine edge cases
| Edge case | Guard |
|---|---|
| **Dreaming starving live calls** | Separate provider from voice (independent free-tier pools) + a hard daily token cap (`DREAM_DAILY_TOKEN_BUDGET`), persisted across restarts, UTC-daily reset. |
| **Groq 429 during a cycle** | Class-level circuit breaker aborts the run; engine backs off **1 hour** (limit resets daily, so retrying every minute just spams 429s). |
| **Control-flow signals swallowed** | `_BudgetExhausted` / `_RateLimitHit` inherit `BaseException` so `except Exception` can't eat them. |
| **Re-analyzing the same turns** | Per-cycle processed markers + `dream_processed` flag (Bug #8). |
| **Customer connects mid-cycle** | Checkpoint saved to Qdrant; resume next idle window. |
| **A cycle crashes** | Caught and logged; the loop continues to the next cycle rather than dying. |
| **Any Qdrant/Groq call fails** | Every cloud op is wrapped in try/except — one failure never crashes the loop. |

### Environment gotcha (documented for maintainers)
- **OneDrive + `uvicorn --reload` is a trap:** file-change events don't fire reliably on the
  OneDrive-synced project folder, so `--reload` silently runs **stale code**. Always fully
  restart the server after a change, and check for duplicate `python.exe`/uvicorn processes
  before debugging a "fix that didn't work."

---

## 8. Architecture Map & Data Stores

**Qdrant Cloud collections** (all created idempotently by `QdrantStore.init_collections()` at
startup):

| Collection | Written by | Read by | Vector |
|---|---|---|---|
| `knowledge_base` | ingestor | RetrievalPipeline | real e5 embeddings (384-d, Cosine) |
| `execution_traces` | live path (`record_turn`) | Dream Engine (offline) | dummy zero-vector |
| `improvement_log` | Dream Engine | `stream_agent` + `/dream/*` | embedded description |
| `customer_profiles` | MemoryConsolidation / session start | `stream_agent` | dummy zero-vector |
| `dream_checkpoints` | Dream cycles + budget ledger | Dream cycles + budget | dummy zero-vector |

**Conversation memory** is separate: LangGraph's in-RAM `MemorySaver`, keyed per-connection by
a UUID `thread_id`. History grows via the `add_messages` reducer and is deleted on disconnect.
It resets on server restart (swap `MemorySaver` for `SqliteSaver`/`PostgresSaver` to persist).

**Graceful degradation:** if `QDRANT_URL`/`QDRANT_API_KEY` are unset or Qdrant init fails, the
store, retrieval pipeline, trace store, and Dream Engine are all set to `None` and the app runs
in **local-only voice mode** (no RAG, no self-improvement) rather than crashing.

---

## 9. Configuration Reference

All configuration lives in `app/config.py` and is env-driven (`.env`). Defaults shown:

| Variable | Default | Purpose |
|---|---|---|
| `SARVAM_API_KEY` | — | Sarvam ASR + TTS (required). |
| `GEMINI_API_KEY` | — | Gemini voice LLM (required). Free at aistudio.google.com — keep on a no-billing project so the free tier stays active. |
| `GEMINI_BASE_URL` | `…/v1beta/openai` | Gemini OpenAI-compatible endpoint base URL. |
| `GROQ_API_KEY` | — | Groq — Dream Engine LLM (required for dreaming). |
| `QDRANT_URL` / `QDRANT_API_KEY` | — | Qdrant Cloud (optional; enables RAG + Dream Engine). |
| `VOICE_LLM_MODEL` | `gemini-3.1-flash-lite` | Live voice agent model (Gemini). |
| `DREAM_LLM_MODEL` | `openai/gpt-oss-120b` | Dream Engine model (Groq). |
| `SARVAM_STT_MODEL` / `SARVAM_STT_MODE` | `saaras:v3` / `transcribe` | Sarvam STT model + mode (`mode` applies only to `saaras:*`). |
| `SARVAM_TTS_MODEL` / `SARVAM_TTS_SPEAKER` | `bulbul:v3` / `simran` | Sarvam TTS model + speaker (speaker must be valid for the model). |
| `DREAM_IDLE_THRESHOLD_SECS` | `300` | Idle time before dreaming starts. |
| `DREAM_CYCLE_INTERVAL_SECS` | `300` | Pause between dream cycles. |
| `DREAM_DAILY_TOKEN_BUDGET` | `200000` | Hard daily token cap for dreaming (full gpt-oss-120b free-tier day). |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | RAG embedding model. |
| `EMBEDDING_DIM` | `384` | Embedding dimension. |
| `EMBEDDING_PROVIDER` | `local` | `local` (sentence-transformers) or `openai`. |
| `RAG_MIN_SCORE` | `0.25` | Minimum similarity score to keep a chunk. |
| `RAG_FILTER_CONFIDENCE_MARGIN` | `0.10` | Triggers the unfiltered retry (Bug #9). |
| `RERANKER` / `RERANKER_MODEL` | `false` / `ms-marco-MiniLM-L-6-v2` | Optional cross-encoder rerank. |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | `false` / — / `DreamSupport` | Optional LangSmith observability. |
| `OPENAI_API_KEY` | — | Only if `EMBEDDING_PROVIDER=openai`. |

---

## 10. Setup, Running & Deployment

### Prerequisites
- Python 3.10+ (3.11 in Docker)
- A modern browser with mic access (Chrome recommended — WebRTC echo cancellation)
- API keys: **Sarvam** (ASR + TTS), **Gemini** (voice LLM), **Groq** (Dream Engine LLM); optional **Qdrant Cloud** for RAG + Dream Engine.

### Local run
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt

# .env with SARVAM_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, (optional) QDRANT_URL/QDRANT_API_KEY
python -m app.knowledge.ingestor   # one-time: load the KB into Qdrant (if using RAG)

# IMPORTANT: do NOT use --reload on the OneDrive-synced folder (runs stale code)
.venv/Scripts/python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000`, click **Start Listening**, and talk.

### Verify a change (per project rules)
```bash
.venv/Scripts/python.exe -m py_compile app/<file>.py    # compiles ≠ works
# For LLM changes, make a real streaming call and inspect the streamed content (+ usage.total_tokens for Groq/dream).
```

### Deployment (Hugging Face Spaces / Docker)
The `Dockerfile` (`python:3.11-slim`) installs CPU-only torch, pre-downloads the **Silero VAD**
model at build time (avoids `NO_SOCKET` at runtime), exposes port **7860**, and runs
`uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}`. Knowledge-base PDFs are **not** baked
into the image — they're ingested into Qdrant Cloud separately and read at runtime. Set
`SARVAM_API_KEY`, `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY` as Space secrets.

> The README YAML frontmatter (`sdk: docker`, `app_port: 7860`) configures the HF Space.

---

## 11. API Reference

### `GET /` — serves the browser UI (`index.html`).

### `GET /health`
```json
{
  "message": "Voice AI Agent Running",
  "qdrant": { "status": "connected | not_configured | error" },
  "dream_engine": { "active_sessions": 0, "is_dreaming": false, "task_alive": true }
}
```

### `GET /dream/knowledge-gaps?limit=50`
Knowledge gaps the Dream Engine found (from `improvement_log`, `category="knowledge_gap"`).
`→ { "count": int, "gaps": [...] }`

### `GET /dream/improvements?category=&limit=50`
All Dream-cycle improvements; optional `category` filter (`prompt` / `retrieval` / `knowledge_gap`).
`→ { "count": int, "improvements": [...] }`

### `POST /voice` (legacy)
Upload a WAV, get a WAV back. One-shot, **no memory** — kept for compatibility.

### `WS /ws` — full-duplex real-time voice

**Client → Server:** binary 16-bit PCM audio · `{"type":"init","sampleRate":48000}` · `{"type":"interrupt"}`

**Server → Client:**
| Message | Meaning |
|---|---|
| `"User: …"` / `"AI: …"` (text) | Transcript lines |
| `{"type":"status","ai_speaking":bool}` | AI speaking state |
| `{"type":"thinking","active":bool}` | Typing indicator |
| `{"type":"language","code":"hi-IN"}` | Detected language badge |
| `{"type":"barge_in"}` | Auto-interrupt (user spoke over AI) |
| `{"type":"interrupted"}` | Manual interrupt acknowledged |
| `{"type":"call_ended"}` | Agent hung up (after goodbye) |
| `{"type":"timeout"}` | Closed after 10 min idle |
| binary WAV | One sentence of AI audio |

---

## 12. Project Structure

```
Voice-AI-Agent/
├── app/
│   ├── main.py               # FastAPI app, WebSocket handler, lifespan wiring
│   ├── config.py             # Single source of truth: models, budgets, thresholds
│   ├── pipecat_pipeline.py   # VAD/STT/TTS processors + VoicePipelineManager
│   ├── langgraph_flow.py     # stream_agent() — the LLM brain (prompt, RAG gate, streaming)
│   ├── memory.py             # LangGraph MemorySaver (per-connection conversation memory)
│   ├── store.py              # QdrantStore wrapper — ALL vector DB ops go through here
│   ├── num_to_words.py       # Digit → spoken words (11 languages) for correct TTS prices
│   │
│   ├── knowledge/            # RAG layer
│   │   ├── embedder.py       #   multilingual-e5-small wrapper (E5 prefixes, LRU cache)
│   │   ├── ingestor.py       #   PDF → structure-aware chunks → Qdrant (run manually)
│   │   └── retriever.py      #   RetrievalPipeline: RAG gate, topic filter, critical injection
│   │
│   ├── dream/                # Self-improvement engine
│   │   ├── engine.py         #   DreamEngine: idle detection, pause/resume, cycle rotation
│   │   ├── cycles.py         #   The 5 cycles + shared LLM helpers
│   │   └── budget.py         #   DreamTokenBudget: hard daily token cap
│   │
│   └── tracing/              # Execution traces + observability
│       ├── trace_store.py    #   ExecutionTraceStore + TurnTrace (feeds the Dream Engine)
│       └── langsmith_setup.py#   Optional LangSmith tracing
│
├── index.html                # Browser UI (mic capture, chat, audio playback)
├── requirements.txt
├── Dockerfile                # HF Spaces / Docker (CPU torch, Silero prebake, port 7860)
├── CLAUDE.md                 # Engineering standing orders for this repo
└── README.md                 # This file
```

---

## 13. Tech Stack

| Layer | Tool |
|---|---|
| Frontend | Vanilla JS + Web Audio API |
| Web framework / server | FastAPI + Uvicorn |
| Voice pipeline | Pipecat (frame-based) |
| VAD | Silero VAD (PyTorch, CPU) |
| ASR | Sarvam `saaras:v3` (`mode=transcribe`) |
| LLM (voice) | Gemini `gemini-3.1-flash-lite` (OpenAI-compatible endpoint) |
| LLM (dream) | Groq `openai/gpt-oss-120b` |
| TTS | Sarvam `bulbul:v3` (speaker `simran`) |
| Agent / memory | LangGraph + MemorySaver |
| Vector DB | Qdrant Cloud |
| Embeddings | `intfloat/multilingual-e5-small` (sentence-transformers, 384-d) |
| PDF ingestion | pdfplumber |
| Observability | LangSmith (optional) |
| Logging | Loguru |
| Deployment | Docker / Hugging Face Spaces (port 7860) |

---

*DreamSupport is a portfolio/educational project. BharatConnect is a fictional telecom; all
plans, prices, and policies are synthetic. The agent never asks for OTPs, card numbers, or CVVs.*
