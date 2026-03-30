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
│  Echo cooldown guard (discards audio 1.5s after AI speaks)
│
▼
Pipecat Pipeline (app/pipecat_pipeline.py)
│
├── VADProcessor          ← Silero neural VAD (detects speech start/end)
├── SarvamSTTService      ← Speech-to-Text via Sarvam API
├── GroqLangGraphProcessor ← LLM (Groq) + Conversation Memory (LangGraph)
├── SarvamTTSService      ← Text-to-Speech via Sarvam API
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
└── Status indicator (listening / processing / AI speaking)
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
| `{"type": "interrupt"}` | User interrupts AI mid-speech |

**Messages received from server:**

| Message | Purpose |
|---------|---------|
| `"User: <text>"` | Display user's transcription |
| `"AI: <text>"` | Display AI's response |
| `{"type": "status", "ai_speaking": true/false}` | Update UI status |
| Binary WAV bytes | Play AI audio |

---

### 2. FastAPI WebSocket Server

**File:** [app/main.py](app/main.py)

- Accepts WebSocket connections at `/ws`
- Creates one `VoicePipelineManager` per connection (isolated pipeline + memory)
- Runs two concurrent async tasks per connection:
  - `receive_loop` — reads PCM audio and control messages from browser
  - `send_loop` — reads output frames from pipeline queue and sends to browser

**Echo Cooldown Guard:**
After the AI finishes speaking, incoming audio is ignored for 1.5 seconds (`POST_AI_COOLDOWN_SECS = 1.5`). This prevents the AI's own voice (picked up by the mic) from being re-transcribed and triggering a loop.

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

#### 3.1 VADProcessor — Voice Activity Detection

Uses **Silero VAD** (a pre-trained neural network) to detect when speech starts and ends.

**How it works:**
1. Resample incoming audio from browser rate (48kHz) → 16kHz (Silero's required rate)
2. Split into 512-sample windows
3. Run Silero's LSTM model on each window → outputs a speech probability (0.0–1.0)
4. Buffer audio while probability > 0.5
5. Detect **speech start** after 3 consecutive high-probability chunks (~0.25s)
6. Detect **speech end** after 8 consecutive low-probability chunks (~0.67s of silence)
7. Hard cap at 80 chunks (~6.7s) to prevent infinite buffering
8. Emit a `SpeechEndFrame` containing the complete buffered utterance as a WAV file
9. Reset Silero's hidden state for the next utterance

**Why Silero VAD?**
Much more robust than simple volume/RMS thresholds. Handles whispers, background noise, and varying microphone hardware. Pre-trained on large speech corpora.

**Key thresholds:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `SPEECH_THRESHOLD` | 0.5 | Silero prob above this = speech |
| `MIN_SPEECH_CHUNKS` | 3 | Chunks needed to confirm speech start |
| `SILENCE_CHUNKS_NEEDED` | 8 | Silent chunks needed to end utterance |
| `MAX_BUFFER_CHUNKS` | 80 | Hard cap (~6.7s) |

---

#### 3.2 SarvamSTTService — Speech to Text

**API:** `POST https://api.sarvam.ai/speech-to-text`

- Receives the `SpeechEndFrame` WAV audio
- Uploads it as multipart form data to Sarvam's ASR API
- Model: `saarika:v2.5` | Language: `en-IN`
- Filters out short noise transcripts and filler words ("yes", "no", "ok", "hmm", etc.)
- Emits a `TranscriptionFrame` (to LLM) and a `TranscriptDisplayFrame` (to browser UI)

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
    messages: Annotated[List[BaseMessage], add_messages]  # grows each turn
    output: str                                            # latest AI reply
```

The `add_messages` reducer **appends** new messages — it never replaces the history. So the full conversation is always available to the LLM.

**Memory persistence:**
- Each WebSocket connection gets a unique `thread_id` (UUID)
- LangGraph's `MemorySaver` stores conversation state per `thread_id`
- On each turn: load state → append user message → call LLM → append AI reply → save state
- Different users never see each other's conversations

**LLM call:**
- Provider: **Groq**
- Model: `llama-3.1-8b-instant`
- Temperature: 0.7
- Max tokens: 120 (~80-100 spoken words — kept short for natural conversation)

---

#### 3.4 SarvamTTSService — Text to Speech

**API:** `POST https://api.sarvam.ai/text-to-speech`

- Receives the AI's response text
- Sends to Sarvam TTS API
- Model: `bulbul:v2` | Speaker: `anushka` | Language: `en-IN`
- Truncates text intelligently at sentence boundaries (Sarvam's 450-char limit)
- Emits:
  - `AIStatusFrame(ai_speaking=True)` → browser shows red pulse
  - `AIAudioFrame(wav_bytes)` → browser plays audio
  - `AIStatusFrame(ai_speaking=False)` → browser resets to listening state

---

#### 3.5 OutputSink

- Receives all frames from the pipeline
- Puts relevant frames (`TranscriptDisplayFrame`, `AIAudioFrame`, `AIStatusFrame`) onto an `asyncio.Queue`
- The WebSocket `send_loop` reads from this queue and delivers to the browser
- Also passes through all Pipecat system frames (`StartFrame`, `StopFrame`, etc.) so the pipeline doesn't hang

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
| Voice Activity Detection | Silero VAD | Neural speech detection |
| ML Runtime | PyTorch + Torchaudio | Silero model inference |
| Speech-to-Text | Sarvam API (`saarika:v2.5`) | Hindi/English ASR |
| Large Language Model | Groq API (`llama-3.1-8b-instant`) | Fast LLM inference |
| Text-to-Speech | Sarvam API (`bulbul:v2`) | Natural-sounding TTS |
| Agent Framework | LangGraph | Stateful conversation graph |
| Conversation Memory | LangGraph MemorySaver | Per-session history checkpointing |
| Message Types | LangChain | `HumanMessage`, `AIMessage`, `BaseMessage` |
| HTTP Client | httpx | Async API calls |
| Env Config | python-dotenv | Load API keys from `.env` |
| Logging | Loguru | Structured, colored logs |
| Deployment | Railway (Nixpacks) | Cloud hosting |

---

## Project Structure

```
Voice-AI-Agent/
│
├── app/
│   ├── main.py               # FastAPI app, WebSocket handler, routes
│   ├── pipecat_pipeline.py   # Full pipeline: VAD → STT → LLM → TTS → OutputSink
│   ├── langgraph_flow.py     # LangGraph agent graph + Groq LLM node
│   ├── memory.py             # MemorySaver instance (shared across connections)
│   ├── config.py             # API key loading from environment
│   ├── asr.py                # Legacy Sarvam ASR helper (not used in WebSocket flow)
│   ├── tts.py                # Legacy Sarvam TTS helper (not used in WebSocket flow)
│   ├── llm.py                # Legacy Groq helper (no memory, not used in WS flow)
│   └── pipeline.py           # Legacy one-shot pipeline (used by POST /voice)
│
├── index.html                # Browser UI (mic capture, chat display, audio playback)
├── requirements.txt          # Python dependencies
├── railway.toml              # Railway deployment config
├── .env                      # API keys (not committed to git)
└── README.md                 # This file
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- A modern browser with microphone access (Chrome recommended)
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

> **Note:** PyTorch and Torchaudio can be large. If you have a GPU, install the CUDA version of PyTorch for faster Silero VAD inference. CPU works fine for this use case.

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

> **Important:** Microphone access requires either `localhost` or an HTTPS connection. Direct file open (`file://`) works in most browsers for local development.

### Start a conversation

1. Click **"Start Listening"**
2. Speak into your microphone
3. Wait for the status to change from "Listening..." to "Processing..."
4. The AI will respond with audio and the transcript will appear in the chat
5. When the AI finishes, the status returns to "Listening..." — speak again
6. Click **"Stop"** to end the session
7. Click **"⚡ Interrupt AI"** to stop the AI mid-response

---

## Deployment on Railway

The project includes a [railway.toml](railway.toml) for one-click deployment:

```toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "uvicorn app.main:app --host 0.0.0.0 --port $PORT"
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 3
```

Set the environment variables `SARVAM_API_KEY` and `GROQ_API_KEY` in the Railway dashboard before deploying.

---

## Key Design Decisions

### Why Silero VAD instead of energy-based detection?
Simple volume thresholds break with background noise, quiet speakers, or different microphone hardware. Silero is a pre-trained neural network that outputs a speech probability score — far more robust.

### Why Pipecat?
Pipecat's frame-based pipeline makes each component (VAD, STT, LLM, TTS) independently replaceable. Swap Sarvam for Deepgram, or Groq for OpenAI, by changing one class.

### Why LangGraph for memory?
LangGraph's checkpointing system handles all the state management automatically. Each WebSocket connection gets a unique `thread_id`, and the `add_messages` reducer ensures history only grows — never accidentally replaced. Upgrading from in-memory to database-backed persistence requires changing a single line.

### Why 120 max tokens for the LLM?
Voice responses need to be concise. 120 tokens ≈ 80–100 spoken words ≈ 20–30 seconds of speech. Long responses feel unnatural in voice conversations.

### Why a 1.5s echo cooldown?
Without it, the AI's own voice is captured by the microphone, transcribed as user input, and triggers another AI response — creating an infinite loop. The cooldown discards audio immediately after the AI speaks.

---

## API Reference

### WebSocket `/ws`

**Client → Server:**

| Frame type | Content | Description |
|------------|---------|-------------|
| Binary | Raw 16-bit PCM bytes | Microphone audio |
| Text JSON | `{"type": "init", "sampleRate": N}` | Report browser sample rate |
| Text JSON | `{"type": "interrupt"}` | Interrupt AI speech |

**Server → Client:**

| Frame type | Content | Description |
|------------|---------|-------------|
| Text | `"User: <transcript>"` | User's transcribed speech |
| Text | `"AI: <response>"` | AI's text response |
| Text JSON | `{"type": "status", "ai_speaking": true}` | AI started speaking |
| Text JSON | `{"type": "status", "ai_speaking": false}` | AI finished speaking |
| Text JSON | `{"type": "interrupted"}` | Interrupt acknowledged |
| Binary | WAV audio bytes | AI's synthesized voice |

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
pipecat-ai>=0.0.60
silero-vad>=5.1.2
torch>=2.0.0
torchaudio>=2.0.0
langgraph>=0.2.0
langchain>=0.3.0
loguru>=0.7.0
```
