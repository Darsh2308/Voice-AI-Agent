"""
embedder.py — Multilingual text embedding + legacy KnowledgeBaseLoader  (Phase 2)
==================================================================================

Embedding model: intfloat/multilingual-e5-small
  - 384-dim native (no padding needed)
  - 100+ languages including Hindi, Marathi, Tamil, Telugu, Kannada, Bengali
  - ~120 MB, fast on CPU
  - Requires query/passage prefixes (E5 paper convention):
      index time : embed("passage: " + chunk_text)
      query time : embed("query: "   + query_text)   ← done in retriever.py

Public API
----------
  embed_text(text, prefix="passage")   — returns 384-d float list
  KnowledgeBaseLoader                  — legacy loader (PDF/TXT/JSON via pypdf)
                                         still works but lacks the smart chunking
                                         of ingestor.py; kept for backward compat

Embedding cache
---------------
  An in-process LRU dict (max 512 entries) avoids re-encoding repeated queries.
  Keyed by (prefix, text). Cache is warm across turns within one server process.
  The voice pipeline benefits most: common questions like "show me prepaid plans"
  embed once then hit cache on every subsequent call.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from loguru import logger

from app.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_URL,
)
from app.store import KNOWLEDGE_BASE, QdrantStore

# ---------------------------------------------------------------------------
# Chunking constants (used by legacy KnowledgeBaseLoader only)
# ---------------------------------------------------------------------------

CHUNK_TOKENS        = 512
CHUNK_OVERLAP       = 64
AVG_CHARS_PER_TOKEN = 4

CHUNK_SIZE   = CHUNK_TOKENS  * AVG_CHARS_PER_TOKEN   # ~2048 chars
OVERLAP_SIZE = CHUNK_OVERLAP * AVG_CHARS_PER_TOKEN   # ~256 chars
UPSERT_BATCH = 32


# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------

_local_model = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        model_name = EMBEDDING_MODEL  # intfloat/multilingual-e5-small
        logger.info(f"Loading embedding model '{model_name}' …")
        _local_model = SentenceTransformer(model_name)
        logger.info(f"Embedding model loaded ✓  dim={_local_model.get_sentence_embedding_dimension()}")
    return _local_model


# ---------------------------------------------------------------------------
# Embedding cache  (in-process LRU, max 512 entries)
# ---------------------------------------------------------------------------

_embed_cache: dict[tuple[str, str], list[float]] = {}
_CACHE_MAX = 512


def _cache_get(prefix: str, text: str) -> list[float] | None:
    return _embed_cache.get((prefix, text))


def _cache_put(prefix: str, text: str, vector: list[float]) -> None:
    if len(_embed_cache) >= _CACHE_MAX:
        # Evict oldest entry (insertion order, Python 3.7+)
        _embed_cache.pop(next(iter(_embed_cache)))
    _embed_cache[(prefix, text)] = vector


# ---------------------------------------------------------------------------
# Public: embed_text()
# ---------------------------------------------------------------------------

async def embed_text(
    text: str,
    prefix: Literal["passage", "query"] = "passage",
) -> list[float]:
    """
    Embed a single string and return a 384-d float list.

    prefix:
      "passage" — use for chunks at index time  (default)
      "query"   — use for user queries at search time

    The E5 model family requires these prefixes to align query/passage
    representations correctly.  Omitting them silently degrades recall.

    Results are cached in-process — repeated identical (prefix, text) pairs
    return instantly without re-encoding.
    """
    if not text or not text.strip():
        return [0.0] * EMBEDDING_DIM

    # Cache check
    cached = _cache_get(prefix, text)
    if cached is not None:
        return cached

    provider = EMBEDDING_PROVIDER.lower()
    if provider == "openai":
        vector = await _embed_openai(text)
    else:
        vector = await _embed_local(text, prefix)

    _cache_put(prefix, text, vector)
    return vector


async def _embed_local(text: str, prefix: str) -> list[float]:
    """Run sentence-transformers encode in a thread pool (non-blocking)."""
    loop  = asyncio.get_event_loop()
    model = _get_local_model()
    # E5 prefix convention
    prefixed = f"{prefix}: {text}"
    vector = await loop.run_in_executor(
        None,
        lambda: model.encode(prefixed, normalize_embeddings=True).tolist(),
    )
    # Trim/pad to EMBEDDING_DIM in case model dim differs from config
    if len(vector) > EMBEDDING_DIM:
        vector = vector[:EMBEDDING_DIM]
    elif len(vector) < EMBEDDING_DIM:
        vector = vector + [0.0] * (EMBEDDING_DIM - len(vector))
    return vector


async def _embed_openai(text: str) -> list[float]:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set but EMBEDDING_PROVIDER=openai")
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={"model": "text-embedding-3-small", "input": text, "dimensions": EMBEDDING_DIM},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


# ---------------------------------------------------------------------------
# Internal dataclass
# ---------------------------------------------------------------------------

@dataclass
class EmbeddedChunk:
    doc_id:      str
    content:     str
    source:      str
    chunk_index: int
    metadata:    dict = field(default_factory=dict)
    vector:      list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# KnowledgeBaseLoader  (legacy — use ingestor.py for BharatConnect PDFs)
# ---------------------------------------------------------------------------

class KnowledgeBaseLoader:
    """
    Generic document loader.  Supports PDF (pypdf), TXT/MD, FAQ JSON.

    For the BharatConnect knowledge base use `app/knowledge/ingestor.py`
    instead — it is layout-aware and produces richer chunk metadata.

    This loader remains for loading supplementary documents or custom content.
    """

    def __init__(self, store: QdrantStore | None = None) -> None:
        self._store = store

    async def _get_store(self) -> QdrantStore:
        if self._store is not None:
            return self._store
        s = QdrantStore(QDRANT_URL, QDRANT_API_KEY)
        await s.init_collections()
        return s

    async def load_pdf(self, path: str) -> int:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf is required: pip install pypdf")
        reader = PdfReader(path)
        pages  = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {i+1}]\n{text}")
        full_text = "\n\n".join(pages)
        source    = Path(path).name
        logger.info(f"PDF '{source}': {len(pages)} pages, {len(full_text)} chars")
        return await self._ingest(full_text, source, metadata={"type": "pdf", "pages": len(pages)})

    async def load_text(self, path: str) -> int:
        text   = Path(path).read_text(encoding="utf-8", errors="replace")
        source = Path(path).name
        logger.info(f"Text '{source}': {len(text)} chars")
        return await self._ingest(text, source, metadata={"type": "text"})

    async def load_faq_json(self, path: str) -> int:
        raw    = json.loads(Path(path).read_text(encoding="utf-8"))
        source = Path(path).name
        if not isinstance(raw, list):
            raise ValueError(f"{path}: expected a JSON list of {{question, answer}} objects")
        chunks_text = []
        for item in raw:
            q = item.get("question", "").strip()
            a = item.get("answer",   "").strip()
            if q and a:
                chunks_text.append(f"Q: {q}\nA: {a}")
        logger.info(f"FAQ '{source}': {len(chunks_text)} Q&A pairs")
        return await self._ingest_pre_chunked(chunks_text, source, metadata={"type": "faq"})

    async def _ingest(self, text: str, source: str, metadata: dict) -> int:
        return await self._ingest_pre_chunked(_split_text(text), source, metadata)

    async def _ingest_pre_chunked(
        self,
        chunks: list[str],
        source: str,
        metadata: dict,
    ) -> int:
        store = await self._get_store()
        total = 0
        batch: list[dict] = []

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            # Use "passage" prefix for index-time embedding
            vector = await embed_text(chunk, prefix="passage")
            doc_id = _content_hash(chunk)

            point = {
                "id":     doc_id,
                "vector": vector,
                "payload": {
                    "doc_id":      doc_id,
                    "content":     chunk,
                    "source":      source,
                    "chunk_index": i,
                    "chunk_type":  "prose",
                    "topic":       "general",
                    "priority":    "normal",
                    "language":    "en",
                    "metadata":    metadata,
                },
            }
            batch.append(point)

            if len(batch) >= UPSERT_BATCH:
                await store.upsert(KNOWLEDGE_BASE, batch)
                total += len(batch)
                batch = []

        if batch:
            await store.upsert(KNOWLEDGE_BASE, batch)
            total += len(batch)

        logger.info(f"Loaded '{source}': {total} chunks upserted into knowledge_base ✓")
        return total


# ---------------------------------------------------------------------------
# Text splitting (legacy, used by KnowledgeBaseLoader)
# ---------------------------------------------------------------------------

def _split_text(text: str) -> list[str]:
    text   = text.strip()
    chunks = []
    start  = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        if end < len(text):
            para_break = text.rfind("\n\n", start, end)
            if para_break != -1 and para_break > start:
                end = para_break
            else:
                for punc in ".!?।":
                    sent_break = text.rfind(punc, start, end)
                    if sent_break != -1 and sent_break > start:
                        end = sent_break + 1
                        break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = max(end - OVERLAP_SIZE, end) if end >= len(text) else end - OVERLAP_SIZE
        if start <= 0 or start >= len(text):
            break

    return chunks


def _content_hash(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return str(uuid.UUID(digest[:32]))


# ---------------------------------------------------------------------------
# CLI entry point (legacy loader)
# ---------------------------------------------------------------------------

async def _cli_main():
    parser = argparse.ArgumentParser(
        description="Load documents into the DreamSupport knowledge base (legacy loader)"
    )
    parser.add_argument("--file", required=True, help="Path to PDF, TXT, MD, or FAQ JSON")
    args   = parser.parse_args()
    path   = Path(args.file)
    ext    = path.suffix.lower()
    loader = KnowledgeBaseLoader()

    if ext == ".pdf":
        count = await loader.load_pdf(str(path))
    elif ext == ".json":
        count = await loader.load_faq_json(str(path))
    elif ext in {".txt", ".md"}:
        count = await loader.load_text(str(path))
    else:
        print(f"Unsupported file type: {ext}. Use .pdf, .txt, .md, or .json")
        return

    print(f"\n✓ Loaded {count} chunks from '{path.name}' into knowledge_base.")


if __name__ == "__main__":
    asyncio.run(_cli_main())
