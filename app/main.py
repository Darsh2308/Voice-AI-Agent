"""
main.py — WebSocket server with all enhanced features
=======================================================

Changes from the original Phase 7 version:

Feature 1  – Interrupt Handling Fix
  After manager.interrupt() is called, the output_queue is already drained
  inside VoicePipelineManager. Here we set is_ai_speaking=False and
  last_ai_finished_at=0.0 immediately so receive_loop stops blocking audio.

Feature 2  – Connection Timeout
  timeout_watch() coroutine checks inactivity every 60 seconds. If no audio
  has arrived for 10 minutes it sends a {"type":"timeout"} message, closes
  the WebSocket, and lets the finally block clean up the pipeline.

Feature 4  – Barge-in
  Echo cooldown is reduced to 0.3 s (WebRTC echo cancellation in the browser
  handles the rest). Audio is ALWAYS forwarded to the pipeline — including
  while the AI is speaking. When VAD detects speech during AI playback it
  emits BargeInDetectedFrame, which send_loop catches to auto-interrupt.
  manager.set_barge_in_mode() tells the VAD when to watch for barge-in.

Feature 6  – Typing Indicator
  AIThinkingFrame(True/False) from GroqLangGraphProcessor is forwarded to the
  browser as {"type":"thinking","active":true/false}.

Feature 7  – Language Auto-Switch
  LanguageDetectedFrame from SarvamSTTService is forwarded to the browser as
  {"type":"language","code":"hi-IN"} so the UI can show a language badge.
"""

import asyncio
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="DEBUG")

from app.pipecat_pipeline import (
    VoicePipelineManager,
    AIAudioFrame,
    AIStatusFrame,
    AIThinkingFrame,          # Feature 6
    LanguageDetectedFrame,    # Feature 7
    BargeInDetectedFrame,     # Feature 4
    EndCallFrame,             # Agent-initiated hangup
    TranscriptDisplayFrame,
    EndFrame,
)
from app.store import QdrantStore
from app.config import QDRANT_URL, QDRANT_API_KEY
from app.langgraph_flow import set_retrieval_pipeline, set_qdrant_store
from app.observability import init_langsmith
from groq import AsyncGroq
from app.config import GROQ_API_KEY


# ── Application lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialise Qdrant store and create all 5 collections.
    Shutdown: close the Qdrant HTTP client cleanly.

    app.state.store is available to all request handlers after startup.
    Qdrant init is skipped gracefully when QDRANT_URL is not configured so
    the app still starts in local-only mode during development.
    """
    # Phase 6: shared Groq client (reused by DreamEngine cycles)
    groq_client = AsyncGroq(api_key=GROQ_API_KEY)

    if QDRANT_URL and QDRANT_API_KEY:
        try:
            store = QdrantStore(QDRANT_URL, QDRANT_API_KEY)
            await store.init_collections()
            app.state.store = store
            logger.info("Qdrant store ready ✓")

            # Phase 2: wire RAG retrieval pipeline
            from app.knowledge.retriever import RetrievalPipeline
            retrieval_pipeline = RetrievalPipeline(store)
            app.state.retrieval_pipeline = retrieval_pipeline
            set_retrieval_pipeline(retrieval_pipeline)
            set_qdrant_store(store)   # Phase 6: allow stream_agent to load dream addenda

            # Pre-warm embedding model at startup so the first voice turn
            # doesn't hit a cold model-load (which takes ~7s and causes RAG timeout)
            try:
                from app.knowledge.embedder import embed_text
                await embed_text("warmup", prefix="query")
                logger.info("Embedding model pre-warmed ✓")
            except Exception as _warm_exc:
                logger.warning(f"Embedding warmup failed (non-fatal): {_warm_exc}")

            # Phase 3: wire execution trace store
            from app.tracing.trace_store import ExecutionTraceStore
            trace_store = ExecutionTraceStore(store)
            app.state.trace_store = trace_store
            logger.info("ExecutionTraceStore ready ✓")

            # Phase 5+6: Dream Engine — starts paused, wakes on first idle window
            from app.dream.engine import DreamEngine
            dream_engine = DreamEngine(
                store              = store,
                retrieval_pipeline = retrieval_pipeline,
                trace_store        = trace_store,
                groq_client        = groq_client,
            )
            await dream_engine.start()
            app.state.dream_engine = dream_engine
            logger.info("DreamEngine started ✓")

        except Exception as exc:
            logger.error(f"Qdrant init failed — running without persistent store: {exc}")
            app.state.store             = None
            app.state.retrieval_pipeline = None
            app.state.trace_store        = None
            app.state.dream_engine       = None
    else:
        logger.warning(
            "QDRANT_URL / QDRANT_API_KEY not set — "
            "running without persistent store (set them in .env for Phase 1)"
        )
        app.state.store             = None
        app.state.retrieval_pipeline = None
        app.state.trace_store        = None
        app.state.dream_engine       = None

    # Phase 4: LangSmith — always attempted, non-fatal if key not set
    init_langsmith()

    yield

    # Teardown
    dream_engine = getattr(app.state, "dream_engine", None)
    if dream_engine is not None:
        await dream_engine.stop()
        logger.info("DreamEngine stopped ✓")

    if getattr(app.state, "store", None) is not None:
        await app.state.store.close()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health(request: Request):
    payload: dict = {"message": "Voice AI Agent Running"}
    store: QdrantStore | None = getattr(request.app.state, "store", None)
    if store is not None:
        try:
            qdrant_status = await store.health_check()
            payload["qdrant"] = qdrant_status
        except Exception as exc:
            payload["qdrant"] = {"status": "error", "detail": str(exc)}
    else:
        payload["qdrant"] = {"status": "not_configured"}

    dream_engine = getattr(request.app.state, "dream_engine", None)
    payload["dream_engine"] = dream_engine.status() if dream_engine else {"status": "not_configured"}

    return JSONResponse(content=payload)


@app.get("/dream/knowledge-gaps")
async def dream_knowledge_gaps(request: Request, limit: int = 50):
    """
    Phase 6: Return knowledge gaps discovered by the Dream Engine.

    These are entries in improvement_log with category="knowledge_gap".
    Use this endpoint to find out what to add to the knowledge base next.
    """
    store: QdrantStore | None = getattr(request.app.state, "store", None)
    if store is None:
        return JSONResponse(
            content={"error": "Qdrant store not configured"},
            status_code=503,
        )
    try:
        from app.store import IMPROVEMENT_LOG
        records, _ = await store.scroll(
            IMPROVEMENT_LOG,
            filter=store.filter_eq("category", "knowledge_gap"),
            limit=limit,
        )
        gaps = [
            {
                "description":   r["payload"].get("improvement_desc", ""),
                "user_input":    r["payload"].get("user_input", ""),
                "suggested_doc": r["payload"].get("suggested_doc", ""),
                "better_query":  r["payload"].get("better_query", ""),
                "discovered_at": r["payload"].get("applied_at", ""),
            }
            for r in records
        ]
        return JSONResponse(content={"count": len(gaps), "gaps": gaps})
    except Exception as exc:
        return JSONResponse(
            content={"error": str(exc)},
            status_code=500,
        )


@app.get("/dream/improvements")
async def dream_improvements(request: Request, category: str = "", limit: int = 50):
    """
    Phase 6: Return all Dream Cycle improvement log entries.

    Optional ?category= filter: "prompt", "retrieval", "knowledge_gap".
    """
    store: QdrantStore | None = getattr(request.app.state, "store", None)
    if store is None:
        return JSONResponse(
            content={"error": "Qdrant store not configured"},
            status_code=503,
        )
    try:
        from app.store import IMPROVEMENT_LOG
        filt = store.filter_eq("category", category) if category else None
        records, _ = await store.scroll(IMPROVEMENT_LOG, filter=filt, limit=limit)
        items = [
            {
                "category":    r["payload"].get("category", ""),
                "description": r["payload"].get("improvement_desc", ""),
                "approved":    r["payload"].get("approved", False),
                "judge_score": r["payload"].get("judge_score"),
                "applied_at":  r["payload"].get("applied_at", ""),
            }
            for r in records
        ]
        return JSONResponse(content={"count": len(items), "improvements": items})
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.get("/")
async def root():
    # index.html was removed from the repo; FileResponse on a missing path raises
    # RuntimeError → HTTP 500 (Bug #18). Guard and return a clear 404 instead so
    # the root fails cleanly. The voice pipeline lives at /ws and is unaffected.
    index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index.html")
    if not os.path.exists(index_path):
        return JSONResponse(
            content={
                "service": "BharatConnect Voice AI",
                "status": "ok",
                "detail": "No frontend bundled here. Connect a client to the /ws WebSocket endpoint.",
            },
            status_code=404,
        )
    return FileResponse(index_path, media_type="text/html")


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_BROWSER_SAMPLE_RATE = 48000

# Feature 4: reduced from 1.5 s to 0.3 s because WebRTC echo cancellation
# (enabled in the browser) handles the bulk of mic-pickup-of-speaker echo.
POST_AI_COOLDOWN_SECS = 0.3

# Feature 2: close the connection after this many seconds of silence
INACTIVITY_TIMEOUT_SECS = 600   # 10 minutes

# Agent-initiated hangup: after the goodbye audio bytes are sent, we wait for the
# browser to finish PLAYING them before closing, so the farewell is never cut off.
# The wait is computed from the actual goodbye audio length:
#   Sarvam TTS returns WAV @ 22050 Hz, 16-bit, mono = 44,100 bytes/sec of audio.
SARVAM_AUDIO_BYTES_PER_SEC      = 22050 * 2          # sample_rate * bytes_per_sample
AGENT_HANGUP_SAFETY_MARGIN_SECS = 1.0               # network + browser buffer slack
MAX_AGENT_HANGUP_GRACE_SECS     = 12.0              # hard ceiling, just in case


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, request: Request = None):
    """
    Full-duplex real-time voice agent.

    Three coroutines run concurrently (asyncio.gather):
      receive_loop()   — reads PCM audio + control messages from browser
      send_loop()      — reads pipeline output frames, sends to browser
      timeout_watch()  — closes idle connections after INACTIVITY_TIMEOUT_SECS

    Phase 3: generates a session_id per connection, calls begin_session /
    end_session on the ExecutionTraceStore, and passes trace_store into
    VoicePipelineManager so every turn is recorded automatically.
    """
    await websocket.accept()

    # Phase 3: per-connection session identity
    session_id   = str(uuid.uuid4())
    trace_store  = getattr(websocket.app.state, "trace_store",  None)
    dream_engine = getattr(websocket.app.state, "dream_engine", None)

    logger.info(f"WebSocket client connected — session_id={session_id[:8]}…")

    # Phase 6: pause the Dream Engine immediately (customer is now active)
    if dream_engine is not None:
        dream_engine.customer_connected()

    # Begin session in Qdrant (non-fatal if store unavailable)
    if trace_store is not None:
        try:
            await trace_store.begin_session(session_id, customer_id="anonymous", language="unknown")
        except Exception as exc:
            logger.warning(f"begin_session failed (non-fatal): {exc}")

    # ── Per-connection state ──────────────────────────────────────────────────
    browser_sample_rate = DEFAULT_BROWSER_SAMPLE_RATE
    last_ai_finished_at = 0.0
    is_ai_speaking      = False
    last_activity       = time.time()
    # Feature 4 (barge-in) — playback-aware window.
    # The server finishes SENDING audio bytes in a burst (~ms), but the browser
    # keeps PLAYING them for seconds. barge-in must stay armed for the whole
    # playback, not just the send. We estimate playback end from bytes sent and
    # keep barge_in_mode ON until then via this deferred task.
    barge_off_task: asyncio.Task | None = None

    manager = VoicePipelineManager(session_id=session_id, trace_store=trace_store)
    await manager.start()

    # Agent speaks first — trigger the opening greeting without waiting for the customer
    asyncio.create_task(manager.trigger_greeting(), name="agent-opening-greeting")

    # ── receive_loop ─────────────────────────────────────────────────────────
    async def receive_loop():
        """
        Read frames from the browser:
          - Binary frames  → raw PCM audio → pushed into pipeline after cooldown
          - Text frames    → JSON control messages (init, interrupt)

        Feature 4 (Barge-in): audio is forwarded to the pipeline even while
        the AI is speaking. The VAD is in barge-in mode during that time and
        will emit BargeInDetectedFrame when it detects the user speaking,
        which send_loop catches and uses to interrupt.

        The cooldown (0.3 s) only prevents the very first chunks after the AI
        stops speaking from being picked up (residual speaker echo). WebRTC echo
        cancellation in the browser handles the rest.
        """
        nonlocal browser_sample_rate, last_ai_finished_at, last_activity, is_ai_speaking

        try:
            while True:
                message = await websocket.receive()

                # ── Text frame: control message ───────────────────────────
                if "text" in message:
                    try:
                        meta = json.loads(message["text"])

                        if meta.get("type") == "init":
                            browser_sample_rate = int(meta["sampleRate"])
                            manager.update_sample_rate(browser_sample_rate)
                            logger.info(f"Browser sample rate: {browser_sample_rate} Hz")

                        elif meta.get("type") == "interrupt":
                            # Feature 1: interrupt + drain stale frames
                            await manager.interrupt()
                            is_ai_speaking      = False
                            last_ai_finished_at = 0.0    # no cooldown after interrupt
                            manager.set_barge_in_mode(False)   # Feature 4
                            await websocket.send_text(json.dumps({"type": "interrupted"}))
                            logger.info("Manual interrupt received from client")

                    except (json.JSONDecodeError, KeyError):
                        pass
                    continue

                # ── Binary frame: raw PCM audio ───────────────────────────
                pcm_bytes = message.get("bytes", b"")
                if not pcm_bytes:
                    continue

                # Feature 2: record activity time
                last_activity = time.time()

                # Feature 4 (Barge-in) + original echo cooldown:
                # Apply a SHORT cooldown (0.3 s) only after the AI finishes a
                # sentence. During this tiny window the last few frames of
                # speaker audio may not yet be cancelled by WebRTC EC.
                # Outside this window, audio always flows — even during AI speech
                # (that's what enables barge-in).
                elapsed = time.time() - last_ai_finished_at
                if elapsed < POST_AI_COOLDOWN_SECS:
                    continue

                await manager.push_audio(pcm_bytes, sample_rate=browser_sample_rate)

        except Exception as e:
            logger.info(f"receive_loop ended: {e}")

    # ── send_loop ─────────────────────────────────────────────────────────────
    async def send_loop():
        """
        Read output frames from the pipeline and send them to the browser.

        Frame types:
          TranscriptDisplayFrame  → "User: …" / "AI: …" chat lines
          AIStatusFrame           → {"type":"status","ai_speaking":bool}
          AIAudioFrame            → raw WAV bytes for playback
          AIThinkingFrame         → Feature 6: {"type":"thinking","active":bool}
          LanguageDetectedFrame   → Feature 7: {"type":"language","code":"hi-IN"}
          BargeInDetectedFrame    → Feature 4: auto-interrupt during playback
          EndFrame                → pipeline shutdown — exit loop
        """
        nonlocal last_ai_finished_at, is_ai_speaking, barge_off_task

        # Track audio bytes delivered in the CURRENT speaking turn so that, on an
        # agent-initiated hangup, we can wait for the browser to actually finish
        # PLAYING the goodbye before closing the socket (otherwise the farewell
        # gets cut off mid-sentence). Reset each time the AI starts a new turn.
        turn_audio_bytes = 0

        while True:
            try:
                frame = await asyncio.wait_for(
                    manager.output_queue.get(),
                    timeout=60.0,
                )
            except asyncio.TimeoutError:
                continue

            # ── TranscriptDisplayFrame ────────────────────────────────────
            if isinstance(frame, TranscriptDisplayFrame):
                prefix = "User" if frame.speaker == "user" else "AI"
                await websocket.send_text(f"{prefix}: {frame.text}")

            # ── AIStatusFrame ─────────────────────────────────────────────
            elif isinstance(frame, AIStatusFrame):
                if frame.ai_speaking and not is_ai_speaking:
                    # AI just started speaking — arm barge-in immediately, and
                    # cancel any pending "turn barge-in off" from a prior turn.
                    is_ai_speaking = True
                    turn_audio_bytes = 0           # new turn — reset audio tally
                    if barge_off_task and not barge_off_task.done():
                        barge_off_task.cancel()
                        barge_off_task = None
                    manager.set_barge_in_mode(True)   # Feature 4

                elif not frame.ai_speaking and is_ai_speaking:
                    # The server has finished SENDING audio, but the browser is
                    # still PLAYING it for ~(bytes / bytes_per_sec) seconds. Keep
                    # barge-in armed for that whole window so the user can talk
                    # over the AI and be heard — turning it off now (the old bug)
                    # meant speech during playback was ignored until the audio
                    # finished. Schedule the disarm at estimated playback end.
                    is_ai_speaking = False
                    play_secs = turn_audio_bytes / SARVAM_AUDIO_BYTES_PER_SEC
                    remaining = max(0.0, play_secs)

                    async def _disarm_after(delay: float):
                        nonlocal last_ai_finished_at
                        try:
                            await asyncio.sleep(delay)
                            manager.set_barge_in_mode(False)
                            last_ai_finished_at = time.time()
                        except asyncio.CancelledError:
                            pass

                    if barge_off_task and not barge_off_task.done():
                        barge_off_task.cancel()
                    barge_off_task = asyncio.create_task(
                        _disarm_after(remaining), name="barge-in-disarm"
                    )

                await websocket.send_text(
                    json.dumps({"type": "status", "ai_speaking": frame.ai_speaking})
                )

            # ── AIAudioFrame ──────────────────────────────────────────────
            elif isinstance(frame, AIAudioFrame):
                # Feature 3: multiple WAV chunks arrive (one per sentence).
                # Browser queues and plays them sequentially.
                await websocket.send_bytes(frame.audio_bytes)
                turn_audio_bytes += len(frame.audio_bytes)
                logger.info(f"Sent {len(frame.audio_bytes)} audio bytes to client")

            # ── Feature 6: AIThinkingFrame ────────────────────────────────
            elif isinstance(frame, AIThinkingFrame):
                await websocket.send_text(
                    json.dumps({"type": "thinking", "active": frame.thinking})
                )

            # ── Feature 7: LanguageDetectedFrame ─────────────────────────
            elif isinstance(frame, LanguageDetectedFrame):
                await websocket.send_text(
                    json.dumps({"type": "language", "code": frame.language_code})
                )

            # ── Feature 4: BargeInDetectedFrame ──────────────────────────
            elif isinstance(frame, BargeInDetectedFrame):
                # Fire whenever barge-in is ARMED — not gated on is_ai_speaking,
                # which flips False as soon as the server finishes SENDING audio
                # even though the browser is still playing it. Gating on it here
                # was part of why mid-playback barge-in was missed.
                logger.info("Barge-in detected — interrupting AI")
                # Cancel any pending playback-end disarm so it can't race us.
                if barge_off_task and not barge_off_task.done():
                    barge_off_task.cancel()
                    barge_off_task = None
                # Feature 1: interrupt + drain stale frames from queue
                await manager.interrupt()
                is_ai_speaking      = False
                last_ai_finished_at = 0.0    # no cooldown — user is already speaking
                manager.set_barge_in_mode(False)
                # Tell browser to stop current audio and show "Listening"
                await websocket.send_text(json.dumps({"type": "barge_in"}))

            # ── EndCallFrame: agent-initiated hangup ──────────────────────
            elif isinstance(frame, EndCallFrame):
                # The goodbye audio bytes have been SENT, but the browser still
                # needs wall-clock time to PLAY them. A fixed grace cut off longer
                # farewells, so compute the real playback duration from the bytes
                # delivered this turn and wait that long (+ a small safety margin).
                play_secs = turn_audio_bytes / SARVAM_AUDIO_BYTES_PER_SEC
                grace = min(
                    MAX_AGENT_HANGUP_GRACE_SECS,
                    play_secs + AGENT_HANGUP_SAFETY_MARGIN_SECS,
                )
                logger.info(
                    f"Agent-initiated end-of-call — goodbye ≈{play_secs:.1f}s "
                    f"({turn_audio_bytes} bytes), waiting {grace:.1f}s before closing"
                )
                try:
                    await websocket.send_text(json.dumps({"type": "call_ended"}))
                except Exception:
                    pass
                await asyncio.sleep(grace)
                try:
                    await websocket.close()
                except Exception:
                    pass
                logger.info("Agent hung up the call — send_loop exiting")
                break

            # ── EndFrame ──────────────────────────────────────────────────
            elif isinstance(frame, EndFrame):
                logger.info("Pipeline EndFrame received — send_loop exiting")
                break

    # ── Feature 2: timeout_watch ─────────────────────────────────────────────
    async def timeout_watch():
        """
        Feature 2: Close idle WebSocket connections after 10 minutes.
        Checks every 60 seconds whether any audio has arrived recently.
        Sends a {"type":"timeout"} message before closing so the browser
        can display a friendly "session ended" message.
        """
        while True:
            await asyncio.sleep(60)
            idle_secs = time.time() - last_activity
            if idle_secs >= INACTIVITY_TIMEOUT_SECS:
                logger.info(f"Connection idle for {idle_secs:.0f}s — closing")
                try:
                    await websocket.send_text(json.dumps({"type": "timeout"}))
                    await websocket.close()
                except Exception:
                    pass
                return

    # ── Run all three coroutines concurrently ─────────────────────────────────
    # Use FIRST_COMPLETED (not gather): the connection is over as soon as ANY of
    # the three loops exits — receive_loop returns on client disconnect, send_loop
    # breaks on agent hangup/EndFrame, timeout_watch returns on idle. gather()
    # waited for ALL three, so a disconnect left send_loop blocked forever on
    # output_queue.get() and a hangup left timeout_watch sleeping up to 10 min —
    # in both cases the finally cleanup (end_session, Dream resume, manager.stop)
    # never ran (leak) or ran ~10 min late (blocked the shared Dream budget window).
    tasks = [
        asyncio.create_task(receive_loop(),  name="ws-receive-loop"),
        asyncio.create_task(send_loop(),     name="ws-send-loop"),
        asyncio.create_task(timeout_watch(), name="ws-timeout-watch"),
    ]
    try:
        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED
        )
        # Cancel the siblings that are still running so cleanup runs immediately.
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        # Surface any non-cancellation error from the loop that finished first.
        for task in done:
            exc = task.exception()
            if exc is not None:
                logger.info(f"WebSocket session ended: {exc}")
    except Exception as e:
        logger.info(f"WebSocket session ended: {e}")
    finally:
        # Cancel a pending barge-in disarm timer so it doesn't outlive the socket.
        if barge_off_task and not barge_off_task.done():
            barge_off_task.cancel()

        # Phase 3: mark session complete in trace store
        if trace_store is not None:
            try:
                await trace_store.end_session(session_id)
            except Exception as exc:
                logger.warning(f"end_session failed (non-fatal): {exc}")

        # Phase 6: notify Dream Engine that this customer is gone
        if dream_engine is not None:
            dream_engine.customer_disconnected()

        await manager.stop()
        logger.info(f"WebSocket client disconnected — session_id={session_id[:8]}… pipeline cleaned up")
