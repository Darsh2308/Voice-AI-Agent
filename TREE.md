# Project Structure — BharatConnect Voice AI Agent

Generated snapshot of the repository layout. Excludes `.git/`, `.venv/`, `.idea/`,
and `__pycache__/` (build/IDE/env noise).

```
Voice-AI-Agent/
├── .claude/                          # Claude Code project config & skills
│   ├── settings.local.json
│   └── skills/
│       ├── deep-debug/SKILL.md       # Root-cause investigation workflow
│       └── ship-feature/SKILL.md     # Feature implementation workflow
├── app/                              # Application source
│   ├── main.py                       # FastAPI app, WebSocket handler, lifespan wiring
│   ├── config.py                     # Single source of truth: models, budgets, thresholds
│   ├── langgraph_flow.py             # stream_agent() — system prompt, RAG gate, Groq calls
│   ├── pipecat_pipeline.py           # VAD, STT, TTS processors + VoicePipelineManager
│   ├── asr.py                        # Speech recognition
│   ├── tts.py                        # Text-to-speech
│   ├── llm.py                        # LLM client wrapper
│   ├── memory.py                     # Conversation memory
│   ├── pipeline.py                   # (legacy/aux pipeline)
│   ├── store.py                      # Qdrant wrapper — all vector DB ops
│   ├── num_to_words.py               # Number → spoken words for TTS
│   ├── observability.py              # Logging / metrics
│   ├── dream/                        # Offline self-improvement engine
│   │   ├── __init__.py
│   │   ├── engine.py                 # Idle-triggered loop, pause/resume, backoff
│   │   ├── cycles.py                 # The 5 dream cycles + shared LLM helpers
│   │   └── budget.py                 # Hard daily token cap (protects voice budget)
│   ├── knowledge/                    # RAG subsystem
│   │   ├── __init__.py
│   │   ├── retriever.py              # Topic pre-filter, critical-chunk injection, search
│   │   ├── ingestor.py               # PDF → chunks → Qdrant (run manually)
│   │   └── embedder.py               # multilingual-e5-small embeddings
│   └── tracing/
│       ├── __init__.py
│       └── trace_store.py            # Trace persistence
├── data/                             # Knowledge base source PDFs
│   ├── 01_BharatConnect_Company_Overview.pdf
│   ├── 02_BharatConnect_Policies_and_Terms.pdf
│   ├── 03_BharatConnect_Billing_Recharges_Plans.pdf
│   ├── 04_BharatConnect_Network_and_Technology.pdf
│   └── 05_BharatConnect_Competitive_Landscape.pdf
├── services/
│   └── pipecat_pipeline.py           # (legacy/aux service pipeline)
├── CLAUDE.md                         # Project standing orders for Claude Code
├── README.md                         # Project documentation
├── info.md                           # Additional notes
├── DreamSupport.pdf                  # Dream engine reference doc
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Container build
├── .dockerignore
├── railway.toml                      # Railway deployment config
├── .env                              # Local secrets (gitignored)
├── .env.example                      # Env var template
└── .gitignore
```
