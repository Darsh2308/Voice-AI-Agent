# Voice AI Agent

A real-time, full-duplex voice AI agent that lets you have natural voice conversations in your browser. You speak — it listens, understands, remembers, and talks back.

---

## What It Does

- Captures your microphone audio in the browser
- Detects when you start and stop speaking (Voice Activity Detection)
- Transcribes your speech to text (ASR)
- Sends the text to an LLM with full conversation memory
- Converts the AI's reply to speech (TTS)
- Streams the audio back and plays it in your browser
- Remembers the entire conversation across turns (per session)
- Lets you interrupt the AI mid-sentence (barge-in)
- Automatically switches TTS language when you switch languages
- Searches the web in real time for current events and facts
- Shows a typing indicator while the AI is thinking
- Adapts its tone based on detected emotion/urgency
- Summarizes long conversations automatically to stay within token limits
- Closes idle connections after 10 minutes of silence

---

## Architecture Overview

```
BROWSER (index.html)
│
│  PCM audio chunks via WebSocket
│
▼
FastAPI WebSocket Server (app/main.py)
│
│  Echo cooldown guard (discards audio 0.3s after AI speaks)
│  Barge-in detection + interrupt
│  Inactivity timeout (10 min)
│
▼
Pipecat Pipeline (app/pipecat_pipeline.py)
│
├── VADProcessor          ← Silero neural VAD (detects speech start/end + barge-in)
├── SarvamSTTService      ← Speech-to-Text via Sarvam API + language detection
├── GroqLangGraphProcessor ← LLM (Groq) + Conversation Memory (LangGraph)
├── SarvamTTSService      ← Text-to-Speech via Sarvam API (auto language switch)
└── OutputSink            ← Queues output frames for WebSocket delivery
│
▼
WebSocket Send Loop
│
│  Transcripts (text) + AI audio (binary WAV) + status updates (JSON)
│
▼
BROWSER
│
├── Chat UI (user/AI message bubbles)
├── Audio playback (<audio> element)
├── Status indicator (listening / processing / AI speaking)
├── Typing indicator (AI is thinking…)
└── Language badge (detected language)
```

---

## Pipeline Deep Dive

### 1. Browser — Audio Capture & WebSocket

**File:** [index.html](index.html)

The browser uses the Web Audio API to capture microphone input:

- `navigator.mediaDevices.getUserMedia()` opens the mic
- `AudioContext.createScriptProcessor()` processes 4096-sample chunks at the browser's native sample rate (usually 48kHz)
- Each chunk is converted from float32 to 16-bit PCM via `floatTo16BitPCM()`
- Sent as binary WebSocket frames to the server

**Control messages sent (JSON):**

| Message | Purpose |
|---------|---------|
| `{"type": "init", "sampleRate": 48000}` | Tell server the browser's sample rate |
| `{"type": "interrupt"}` | User manually interrupts AI mid-speech |

**Messages received from server:**

| Message | Purpose |
|---------|---------|
| `"User: <text>"` | Display user's transcription |
| `"AI: <text>"` | Display AI's response |
| `{"type": "status", "ai_speaking": true/false}` | Update UI status |
| `{"type": "thinking", "active": true/false}` | Show/hide typing indicator |
| `{"type": "language", "code": "hi-IN"}` | Show detected language badge |
| `{"type": "barge_in"}` | AI interrupted by user barge-in |
| `{"type": "interrupted"}` | Manual interrupt acknowledged |
| `{"type": "timeout"}` | Session closed due to inactivity |
| Binary WAV bytes | Play AI audio |

---

### 2. FastAPI WebSocket Server

**File:** [app/main.py](app/main.py)

- Accepts WebSocket connections at `/ws`
- Creates one `VoicePipelineManager` per connection (isolated pipeline + memory)
- Runs three concurrent async tasks per connection:
  - `receive_loop` — reads PCM audio and control messages from browser
  - `send_loop` — reads output frames from pipeline queue and sends to browser
  - `timeout_watch` — closes idle connections after 10 minutes

**Echo Cooldown Guard:**
After the AI finishes speaking, incoming audio is ignored for 0.3 seconds (`POST_AI_COOLDOWN_SECS = 0.3`). WebRTC echo cancellation in the browser handles the rest. This is much shorter than earlier versions because barge-in now requires audio to flow during AI speech.

**Routes:**

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Health check |
| `/voice` | POST | Legacy one-shot: upload WAV, get WAV back (no memory) |
| `/ws` | WebSocket | Full-duplex real-time voice conversation |

---

### 3. Pipecat Pipeline

**File:** [app/pipecat_pipeline.py](app/pipecat_pipeline.py)

Pipecat is a frame-based pipeline framework. Each processor receives frames, does its job, and emits new frames downstream.

#### Custom Frame Types

| Frame | Emitted by | Purpose |
|-------|-----------|---------|
| `SpeechEndFrame` | VADProcessor | Complete buffered utterance as WAV |
| `TranscriptDisplayFrame` | STT / LLM | Chat line to display in browser |
| `AIAudioFrame` | TTS | WAV bytes for one sentence |
| `AIStatusFrame` | TTS | AI speaking state change |
| `AIThinkingFrame` | LLM | Typing indicator (thinking=True/False) |
| `LanguageDetectedFrame` | STT | Detected language code (e.g. `hi-IN`) |
| `EmotionHintFrame` | VAD / STT | Emotion hint (`neutral`, `hesitant`, `agitated`) |
| `BargeInDetectedFrame` | VAD | User spoke while AI was speaking |

---

#### 3.1 VADProcessor — Voice Activity Detection

Uses **Silero VAD** (a pre-trained neural network) to detect when speech starts and ends.

**How it works:**
1. Resample incoming audio from browser rate (48kHz) → 16kHz (Silero's required rate)
2. Split into 512-sample windows
3. Run Silero's LSTM model on each window → outputs a speech probability (0.0–1.0)
4. Buffer audio while probability > 0.5
5. Detect **speech start** after 3 consecutive high-probability chunks (~0.25s)
6. Detect **speech end** after 8 consecutive low-probability chunks (~0.67s of silence) — or 3 chunks in barge-in mode for faster response
7. Hard cap at 80 chunks (~6.7s) to prevent infinite buffering
8. Emit a `SpeechEndFrame` containing the complete buffered utterance as a WAV file
9. Reset Silero's hidden state for the next utterance

In **barge-in mode** (while AI is speaking), when speech first starts the VAD immediately emits a `BargeInDetectedFrame` so the AI can be interrupted without waiting for a full utterance.

The VAD also tracks per-utterance energy. If the energy exceeds 2× the running baseline, it emits `EmotionHintFrame("agitated")`.

**Why Silero VAD?**
Much more robust than simple volume/RMS thresholds. Handles whispers, background noise, and varying microphone hardware.

**Key thresholds:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `SPEECH_THRESHOLD` | 0.5 | Silero prob above this = speech |
| `MIN_SPEECH_CHUNKS` | 3 | Chunks needed to confirm speech start |
| `SILENCE_CHUNKS_NEEDED` | 8 | Silent chunks needed to end utterance (normal) |
| `SILENCE_CHUNKS_BARGEIN` | 3 | Silent chunks to end utterance during barge-in |
| `MAX_BUFFER_CHUNKS` | 80 | Hard cap (~6.7s) |

---

#### 3.2 SarvamSTTService — Speech to Text

**API:** `POST https://api.sarvam.ai/speech-to-text`

- Receives the `SpeechEndFrame` WAV audio
- Uploads it as multipart form data to Sarvam's ASR API
- Model: `saarika:v2.5` | Language: `en-IN`
- Filters out short noise transcripts and filler words ("yes", "no", "ok", "hmm", etc.)
- Emits a `TranscriptionFrame` (to LLM) and a `TranscriptDisplayFrame` (to browser UI)
- Reads the `language_code` from the Sarvam response and emits a `LanguageDetectedFrame` for downstream auto-switching

---

#### 3.3 GroqLangGraphProcessor — LLM with Memory

**Files:** [app/pipecat_pipeline.py](app/pipecat_pipeline.py), [app/langgraph_flow.py](app/langgraph_flow.py), [app/memory.py](app/memory.py)

This is where the AI "thinks" and remembers.

**LangGraph State Machine:**

```
START
  │
  ▼
[llm_node]  ← Calls Groq API with full conversation history
  │
  ▼
 END
```

**State schema:**
```python
class AgentState(TypedDict):
    messages:   Annotated[List[BaseMessage], add_messages]  # grows each turn
    output:     str                                          # latest AI reply
    turn_count: int                                          # triggers summarization
    summary:    str                                          # rolling summary
```

The `add_messages` reducer **appends** new messages — it never replaces the history.

**Memory persistence:**
- Each WebSocket connection gets a unique `thread_id` (UUID)
- LangGraph's `MemorySaver` stores conversation state per `thread_id`
- On each turn: load state → append user message → call LLM → append AI reply → save state
- Different users never see each other's conversations

**Streaming (Feature 3):**
The processor uses `stream_agent()`, an async generator that yields complete sentences one-by-one as the LLM generates them. Each sentence is immediately sent to TTS so the first audio chunk starts playing before the LLM finishes the full response.

**Typing indicator (Feature 6):**
`AIThinkingFrame(True)` is emitted as soon as user speech is transcribed. `AIThinkingFrame(False)` is emitted once the first sentence is ready for TTS.

**LLM call:**
- Provider: **Groq**
- Model: `llama-3.1-8b-instant`
- Temperature: 0.7
- Max tokens: 200 (first call with tool detection), 150 (follow-up after tool)

---

#### 3.4 SarvamTTSService — Text to Speech

**API:** `POST https://api.sarvam.ai/text-to-speech`

- Receives the AI's response text sentence-by-sentence
- Sends to Sarvam TTS API
- Model: `bulbul:v2` | Speaker: `anushka` | Default language: `en-IN`
- Listens for `LanguageDetectedFrame` and automatically switches `target_language_code`
- Truncates text intelligently at sentence boundaries (Sarvam's 450-char limit)
- Emits:
  - `AIStatusFrame(ai_speaking=True)` → browser shows red pulse
  - `AIAudioFrame(wav_bytes)` → browser plays audio (one frame per sentence)
  - `AIStatusFrame(ai_speaking=False)` → browser resets to listening state

---

#### 3.5 OutputSink

- Receives all frames from the pipeline
- Puts relevant frames (`TranscriptDisplayFrame`, `AIAudioFrame`, `AIStatusFrame`, `AIThinkingFrame`, `LanguageDetectedFrame`, `BargeInDetectedFrame`) onto an `asyncio.Queue`
- The WebSocket `send_loop` reads from this queue and delivers to the browser
- Also passes through all Pipecat system frames (`StartFrame`, `StopFrame`, etc.) so the pipeline doesn't hang

---

## Enhanced Features

### Feature 1 — Interrupt Handling

When the user clicks "Interrupt" or barge-in is detected, `manager.interrupt()` drains the output queue, sets `is_ai_speaking=False`, and resets `last_ai_finished_at=0.0` so audio flows again immediately with no cooldown.

### Feature 2 — Connection Timeout

`timeout_watch()` checks inactivity every 60 seconds. After 10 minutes of silence it sends `{"type":"timeout"}` to the browser and closes the WebSocket cleanly.

### Feature 3 — Streaming TTS

`stream_agent()` in [app/langgraph_flow.py](app/langgraph_flow.py) is an async generator that splits LLM output at sentence boundaries (`.`, `?`, `!`) and yields each sentence immediately. TTS starts on the first sentence while the LLM is still generating the rest, reducing perceived latency.

### Feature 4 — Barge-in

Audio always flows to the pipeline — even while the AI is speaking. The VAD enters barge-in mode (`set_barge_in_mode(True)`) when AI speech starts. If Silero detects human speech during playback it emits `BargeInDetectedFrame`. `send_loop` catches this and immediately interrupts the AI, sending `{"type":"barge_in"}` so the browser stops playback.

### Feature 6 — Typing Indicator

`GroqLangGraphProcessor` emits `AIThinkingFrame(True)` before the LLM call and `AIThinkingFrame(False)` once the first sentence is sent to TTS. The browser shows animated "AI is thinking…" dots during this window.

### Feature 7 — Language Auto-Switch

Sarvam ASR returns a `language_code` field. `SarvamSTTService` wraps this in a `LanguageDetectedFrame` which flows to `SarvamTTSService`. The TTS service automatically switches its output language so the AI responds in the same language the user spoke.

### Feature 8 — Conversation Summarization

After every 20 conversation turns, the oldest messages are summarized into a single paragraph using the LLM (`_summarize_history()`). Only the summary + last 4 messages are sent to the API in future turns, keeping context within token limits and reducing costs. The summary is stored in `AgentState.summary`.

### Feature 9 — Web Search Tool

The LLM is given a `web_search` tool backed by DuckDuckGo (`ddgs` — no API key required). When the user asks about current events, weather, sports scores, or recent facts, the LLM calls the tool automatically. The tool result is added to the prompt and a second streaming call produces the final answer. For non-tool turns only one API call is made.

### Feature 10 — Emotion / Tone Detection

The VAD tracks speech energy per utterance. If energy is > 2× the running baseline, `EmotionHintFrame("agitated")` is emitted. `GroqLangGraphProcessor` receives the hint and appends a tone adjustment to the system prompt:

| Hint | System prompt addition |
|------|----------------------|
| `neutral` | (none) |
| `hesitant` | Be extra clear, patient, and encouraging |
| `agitated` | Respond with a calm, brief, empathetic tone |

---

## Conversation Memory — How It Works

Memory is maintained per WebSocket connection using LangGraph + MemorySaver.

**Example conversation:**

```
Turn 1:
  User: "What is my name?"
  → LangGraph state: [HumanMessage("What is my name?")]
  → Groq: "I don't know your name yet. What is it?"
  → LangGraph state: [HumanMessage(...), AIMessage("I don't know...")]

Turn 2:
  User: "My name is Alice"
  → LangGraph state: [msg1, msg2, HumanMessage("My name is Alice")]
  → Groq: "Nice to meet you, Alice!"
  → LangGraph state: [msg1, msg2, msg3, AIMessage("Nice to meet you...")]

Turn 3:
  User: "What's my name?"
  → LangGraph sees full 4-message history
  → Groq: "Your name is Alice."  ← Memory works!
```

Memory lives in RAM (MemorySaver). On server restart, history is lost. To persist across restarts, swap `MemorySaver` for `SqliteSaver` or `PostgresSaver` — the interface is identical.

---

## Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Frontend | Vanilla JS + Web Audio API | Mic capture, audio playback, WebSocket |
| Web Framework | FastAPI | REST + WebSocket server |
| ASGI Server | Uvicorn | Async HTTP + WebSocket serving |
| Voice Pipeline | Pipecat | Modular frame-based audio pipeline |
| Voice Activity Detection | Silero VAD | Neural speech detection + barge-in |
| ML Runtime | PyTorch + Torchaudio | Silero model inference (CPU) |
| Speech-to-Text | Sarvam API (`saarika:v2.5`) | Hindi/English ASR + language detection |
| Large Language Model | Groq API (`llama-3.1-8b-instant`) | Fast streaming LLM inference |
| Text-to-Speech | Sarvam API (`bulbul:v2`) | Natural-sounding TTS (multi-language) |
| Agent Framework | LangGraph | Stateful conversation graph |
| Conversation Memory | LangGraph MemorySaver | Per-session history checkpointing |
| Message Types | LangChain Core | `HumanMessage`, `AIMessage`, `BaseMessage` |
| Web Search | ddgs (DuckDuckGo) | Real-time web search (no API key) |
| Language Detection | langdetect | Fallback language detection |
| HTTP Client | httpx | Async API calls |
| Env Config | python-dotenv | Load API keys from `.env` |
| Logging | Loguru | Structured, colored logs |
| Deployment | Railway (Dockerfile) | Cloud hosting |

---

## Project Structure

```
Voice-AI-Agent/
│
├── app/
│   ├── main.py               # FastAPI app, WebSocket handler, routes
│   ├── pipecat_pipeline.py   # Full pipeline: VAD → STT → LLM → TTS → OutputSink
│   ├── langgraph_flow.py     # LangGraph agent graph + Groq LLM + streaming + tools
│   ├── memory.py             # MemorySaver instance (shared across connections)
│   ├── config.py             # API key loading from environment
│   ├── asr.py                # Legacy Sarvam ASR helper (not used in WebSocket flow)
│   ├── tts.py                # Legacy Sarvam TTS helper (not used in WebSocket flow)
│   ├── llm.py                # Legacy Groq helper (no memory, not used in WS flow)
│   └── pipeline.py           # Legacy one-shot pipeline (used by POST /voice)
│
├── services/
│   └── pipecat_pipeline.py   # (placeholder — unused)
│
├── index.html                # Browser UI (mic capture, chat display, audio playback)
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker image for Railway deployment
├── railway.toml              # Railway deployment config (Dockerfile builder)
├── .env                      # API keys (not committed to git)
└── README.md                 # This file
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- A modern browser with microphone access (Chrome recommended — required for WebRTC echo cancellation)
- API keys for:
  - [Sarvam AI](https://sarvam.ai) — for ASR and TTS
  - [Groq](https://groq.com) — for LLM inference

### 1. Clone the repository

```bash
git clone <repo-url>
cd Voice-AI-Agent
```

### 2. Create a virtual environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch and Torchaudio can be large. The CPU-only version is sufficient for Silero VAD in this project. For GPU inference, install the CUDA version of PyTorch separately.

### 4. Configure API keys

Create a `.env` file in the project root:

```env
SARVAM_API_KEY=your_sarvam_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

---

## Running the Application

### Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For development with auto-reload:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Open the browser UI

Open [index.html](index.html) directly in your browser, **or** serve it via the server.

If serving from the server, open: `http://localhost:8000`

> **Important:** Microphone access requires either `localhost` or an HTTPS connection. Direct file open (`file://`) works in most browsers for local development. Chrome is recommended for WebRTC echo cancellation support.

### Start a conversation

1. Click **"Start Listening"**
2. Speak into your microphone
3. Wait for the status to change from "Listening..." to "Processing..."
4. The AI will respond with audio and the transcript will appear in the chat
5. When the AI finishes, the status returns to "Listening..." — speak again
6. Click **"Stop"** to end the session
7. Click **"⚡ Interrupt AI"** to stop the AI mid-response
8. Speak while the AI is talking to barge in and interrupt automatically

---

## Deployment on Railway

The project uses a [Dockerfile](Dockerfile) for deployment:

```dockerfile
FROM python:3.11-slim

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg

# CPU-only torch (avoids ~3 GB CUDA wheels)
RUN pip install torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY app/ ./app/
COPY services/ ./services/

# Pre-download Silero VAD model at build time (avoids runtime network calls)
RUN python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', ...)"

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

The [railway.toml](railway.toml) is configured to use this Dockerfile:

```toml
[build]
builder = "dockerfile"
dockerfilePath = "Dockerfile"

[deploy]
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 3
```

Set the environment variables `SARVAM_API_KEY` and `GROQ_API_KEY` in the Railway dashboard before deploying.

> **Why pre-download Silero at build time?** Railway's runtime environment may restrict outbound network access. Building the model into the image avoids `NO_SOCKET` errors on first startup.

---

## Key Design Decisions

### Why Silero VAD instead of energy-based detection?
Simple volume thresholds break with background noise, quiet speakers, or different microphone hardware. Silero is a pre-trained neural network that outputs a speech probability score — far more robust. It also enables barge-in without false positives from background noise.

### Why Pipecat?
Pipecat's frame-based pipeline makes each component (VAD, STT, LLM, TTS) independently replaceable. Swap Sarvam for Deepgram, or Groq for OpenAI, by changing one class.

### Why LangGraph for memory?
LangGraph's checkpointing system handles all the state management automatically. Each WebSocket connection gets a unique `thread_id`, and the `add_messages` reducer ensures history only grows — never accidentally replaced. Upgrading from in-memory to database-backed persistence requires changing a single line.

### Why streaming sentences to TTS instead of the full response?
Waiting for the complete LLM response before starting TTS adds 1–3 seconds of silence. Splitting at sentence boundaries and piping each sentence to TTS immediately reduces perceived latency to under 1 second for the first audio.

### Why 0.3s echo cooldown instead of 1.5s?
Chrome's WebRTC echo cancellation (`echoCancellation: true` on `getUserMedia`) eliminates most mic-pickup-of-speaker echo. The 0.3s cooldown only handles residual frames that haven't been cancelled yet. A shorter cooldown makes barge-in feel more responsive.

### Why DuckDuckGo for web search?
No API key required. The `ddgs` library provides a clean Python interface to DuckDuckGo results. Top-3 results are trimmed to 1000 characters to avoid bloating the LLM context.

### Why summarize at 20 turns?
`llama-3.1-8b-instant` has a limited context window. At 20 turns the message list can exceed 2000 tokens. Summarizing old turns into a paragraph keeps every call under 1000 tokens while preserving long-term context.

---

## API Reference

### WebSocket `/ws`

**Client → Server:**

| Frame type | Content | Description |
|------------|---------|-------------|
| Binary | Raw 16-bit PCM bytes | Microphone audio |
| Text JSON | `{"type": "init", "sampleRate": N}` | Report browser sample rate |
| Text JSON | `{"type": "interrupt"}` | Manually interrupt AI speech |

**Server → Client:**

| Frame type | Content | Description |
|------------|---------|-------------|
| Text | `"User: <transcript>"` | User's transcribed speech |
| Text | `"AI: <response>"` | AI's text response |
| Text JSON | `{"type": "status", "ai_speaking": true}` | AI started speaking |
| Text JSON | `{"type": "status", "ai_speaking": false}` | AI finished speaking |
| Text JSON | `{"type": "thinking", "active": true}` | AI started processing |
| Text JSON | `{"type": "thinking", "active": false}` | AI finished processing |
| Text JSON | `{"type": "language", "code": "hi-IN"}` | Detected input language |
| Text JSON | `{"type": "barge_in"}` | User interrupted AI (auto barge-in) |
| Text JSON | `{"type": "interrupted"}` | Manual interrupt acknowledged |
| Text JSON | `{"type": "timeout"}` | Session closed — 10 min idle |
| Binary | WAV audio bytes | AI's synthesized voice (one frame per sentence) |

---

### POST `/voice` (Legacy)

Upload a WAV file, get a WAV file back. No conversation memory.

```bash
curl -X POST http://localhost:8000/voice \
  -F "audio=@your_audio.wav"
```

Response:
```json
{
  "message": "Processed successfully",
  "output_audio": "path/to/output.wav"
}
```

---

## Dependencies

Full list from [requirements.txt](requirements.txt):

```
fastapi==0.135.2
uvicorn==0.42.0
python-dotenv==1.2.2
httpx==0.28.1
pydantic==2.12.5
groq==1.1.2
python-multipart==0.0.22
websockets==16.0
pipecat-ai>=0.0.60
silero-vad>=5.1.2
torch>=2.0.0
torchaudio>=2.0.0
langgraph>=0.2.0
langchain-core>=0.3.0
loguru>=0.7.0
ddgs>=7.0.0
langdetect>=1.0.9
```
