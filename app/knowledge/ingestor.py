"""
ingestor.py — Smart BharatConnect Knowledge Base Ingestor
==========================================================

Layout-aware ingestion of the 5 BharatConnect PDF documents.
Uses pdfplumber (preserves table cell boundaries) instead of raw pypdf.

Chunking rules
--------------
  Rule 1 — Section prose
    Split on heading boundaries (numbered headings like "1.", "1.1", ALL-CAPS).
    Each heading + its prose = one chunk.  Sub-split at paragraphs if > 400 tokens.
    Prepend breadcrumb: "BharatConnect > {doc_name} > {section}: "

  Rule 2 — Catalogue table rows   (plan/add-on/fibre lookup tables)
    Detected when column headers include plan-code/price/validity/data keywords.
    Each row serialised as a self-contained sentence.
    chunk_type = "table_row"

  Rule 3 — Comparison/matrix tables   (competitor matrix, FUP matrix, troubleshooting)
    Detected when table has a "dimension" / "symptom" / competitor-name column.
    Whole table as one atomic chunk.
    chunk_type = "table_full"

  Rule 4 — Callout / critical policy boxes
    Detected by trigger words (⚠, READ FIRST, CRITICAL, NEVER, IMPORTANT, NOTE).
    One atomic chunk per callout.  Stored TWICE (different breadcrumbs) for
    retrieval redundancy.
    chunk_type = "callout",  priority = "critical"

Metadata per chunk
------------------
  doc_id, source, section, chunk_type, topic, priority, language, content, chunk_index

Document → ID → topic mapping
------------------------------
  01_BharatConnect_Company_Overview.pdf      KB-CORP-001   policy, general
  02_BharatConnect_Policies_and_Terms.pdf    KB-POL-002    policy, billing
  03_BharatConnect_Billing_Recharges_Plans.pdf KB-BILL-003 billing, plans_prepaid, plans_postpaid
  04_BharatConnect_Network_and_Technology.pdf  KB-NET-004  network
  05_BharatConnect_Competitive_Landscape.pdf   KB-COMP-005 competitive

CLI
---
  python -m app.knowledge.ingestor                    # ingest all 5 PDFs
  python -m app.knowledge.ingestor --file data/03_…   # single file
  python -m app.knowledge.ingestor --verify           # show chunk counts per doc
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import QDRANT_API_KEY, QDRANT_URL
from app.knowledge.embedder import embed_text
from app.store import KNOWLEDGE_BASE, QdrantStore

# ---------------------------------------------------------------------------
# Document registry
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent.parent / "data"

DOC_REGISTRY: dict[str, dict[str, Any]] = {
    "01_BharatConnect_Company_Overview.pdf": {
        "doc_id": "KB-CORP-001",
        "doc_name": "Company Overview",
        "topic": "policy",
    },
    "02_BharatConnect_Policies_and_Terms.pdf": {
        "doc_id": "KB-POL-002",
        "doc_name": "Policies and Terms",
        "topic": "policy",
    },
    "03_BharatConnect_Billing_Recharges_Plans.pdf": {
        "doc_id": "KB-BILL-003",
        "doc_name": "Billing Recharges and Plans",
        "topic": "billing",
    },
    "04_BharatConnect_Network_and_Technology.pdf": {
        "doc_id": "KB-NET-004",
        "doc_name": "Network and Technology",
        "topic": "network",
    },
    "05_BharatConnect_Competitive_Landscape.pdf": {
        "doc_id": "KB-COMP-005",
        "doc_name": "Competitive Landscape",
        "topic": "competitive",
    },
}

UPSERT_BATCH = 32

# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    content:    str
    doc_id:     str
    source:     str
    section:    str
    chunk_type: str   # prose | table_row | table_full | callout
    topic:      str
    priority:   str   # normal | critical
    chunk_index: int  = 0
    language:   str   = "en"
    extra:      dict  = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

# Matches: "1. Title", "1.1 Title", "1.1.1 Title", "TITLE IN ALL CAPS"
_HEADING_RE = re.compile(
    r"^(\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{2,80}|[A-Z][A-Z\s&\/\-]{4,60})$",
    re.MULTILINE,
)

# Callout trigger phrases
_CALLOUT_TRIGGERS = {
    "read first", "critical", "never", "important", "note:", "warning",
    "⚠", "must know", "always remember", "agents must", "never ask",
    "only on the website", "do not", "must never",
}


def _is_callout_line(line: str) -> bool:
    low = line.lower().strip()
    return any(trigger in low for trigger in _CALLOUT_TRIGGERS)


# ---------------------------------------------------------------------------
# Table type detection
# ---------------------------------------------------------------------------

_CATALOGUE_HEADER_KEYWORDS = {
    "plan code", "price", "validity", "data", "voice", "sms", "speed",
    "rental", "benefit", "pack", "add-on", "quota", "cost", "rate",
    "download", "upload", "mbps", "gbps",
}

_COMPARISON_HEADER_KEYWORDS = {
    "dimension", "symptom", "scenario", "competitor", "telanova", "speedcell",
    "vistatel", "vs", "compared", "fup", "throttle", "type", "category",
}


def _is_catalogue_table(headers: list[str]) -> bool:
    lowered = {h.lower().strip() for h in headers if h}
    return len(lowered & _CATALOGUE_HEADER_KEYWORDS) >= 2


def _is_comparison_table(headers: list[str]) -> bool:
    lowered = {h.lower().strip() for h in headers if h}
    return bool(lowered & _COMPARISON_HEADER_KEYWORDS)


# ---------------------------------------------------------------------------
# Table row serialisation
# ---------------------------------------------------------------------------

def _serialise_catalogue_row(headers: list[str], row: list[str], breadcrumb: str) -> str:
    """Turn one catalogue table row into a self-contained sentence."""
    parts = []
    for h, v in zip(headers, row):
        h = (h or "").strip()
        v = (v or "").strip()
        if h and v and v not in {"-", "—", "N/A", ""}:
            parts.append(f"{h}: {v}")
    return breadcrumb + "; ".join(parts) + "."


def _serialise_full_table(
    headers: list[str],
    rows: list[list[str]],
    caption: str,
    breadcrumb: str,
) -> str:
    """Serialise an entire comparison/matrix table as a readable block."""
    lines = [breadcrumb + caption, ""]
    header_line = " | ".join(h or "" for h in headers)
    lines.append(header_line)
    lines.append("-" * len(header_line))
    for row in rows:
        lines.append(" | ".join(v or "" for v in row))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prose sub-splitting
# ---------------------------------------------------------------------------

_APPROX_CHARS_PER_TOKEN = 4
_MAX_PROSE_CHARS        = 400 * _APPROX_CHARS_PER_TOKEN   # ~1600 chars
_OVERLAP_CHARS          = 60  * _APPROX_CHARS_PER_TOKEN   # ~240 chars


def _split_prose(text: str, breadcrumb: str) -> list[str]:
    """
    Split a prose block at paragraph boundaries if it exceeds _MAX_PROSE_CHARS.
    Each sub-chunk is prefixed with the section breadcrumb.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= _MAX_PROSE_CHARS:
        return [breadcrumb + text]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 > _MAX_PROSE_CHARS and current:
            chunks.append(breadcrumb + current.strip())
            # overlap: carry last paragraph into next chunk
            current = current[-_OVERLAP_CHARS:].lstrip() + "\n\n" + para
        else:
            current = (current + "\n\n" + para).strip() if current else para

    if current.strip():
        chunks.append(breadcrumb + current.strip())

    return chunks


# ---------------------------------------------------------------------------
# Core extractor: one PDF → list[Chunk]
# ---------------------------------------------------------------------------

def extract_chunks(pdf_path: Path, doc_meta: dict) -> list[Chunk]:
    """
    Extract all chunks from one BharatConnect PDF using pdfplumber.

    Returns a flat list of Chunk objects in document order.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is required: pip install pdfplumber")

    doc_id   = doc_meta["doc_id"]
    doc_name = doc_meta["doc_name"]
    topic    = doc_meta["topic"]
    source   = pdf_path.name
    chunks: list[Chunk] = []
    chunk_idx = 0

    current_section  = "Introduction"
    current_prose    = ""
    pending_callout  = ""

    def _flush_prose():
        nonlocal current_prose, chunk_idx
        text = current_prose.strip()
        if not text:
            return
        breadcrumb = f"BharatConnect > {doc_name} > {current_section}: "
        for sub in _split_prose(text, breadcrumb):
            chunks.append(Chunk(
                content    = sub,
                doc_id     = doc_id,
                source     = source,
                section    = current_section,
                chunk_type = "prose",
                topic      = topic,
                priority   = "normal",
                chunk_index = chunk_idx,
            ))
            chunk_idx += 1
        current_prose = ""

    def _flush_callout():
        nonlocal pending_callout, chunk_idx
        text = pending_callout.strip()
        if not text:
            return
        breadcrumb = f"BharatConnect > {doc_name} > {current_section} [POLICY]: "
        # Store twice for retrieval redundancy
        for i, bc in enumerate([breadcrumb, f"BharatConnect CRITICAL RULE — {doc_name}: "]):
            chunks.append(Chunk(
                content    = bc + text,
                doc_id     = doc_id,
                source     = source,
                section    = current_section,
                chunk_type = "callout",
                topic      = topic,
                priority   = "critical",
                chunk_index = chunk_idx + i,
                extra      = {"duplicate_for_redundancy": i > 0},
            ))
        chunk_idx += 2
        pending_callout = ""

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):

            # ── Process tables on this page first ──────────────────────────
            tables = page.extract_tables() or []
            table_bboxes = [t.bbox for t in page.find_tables()] if tables else []

            for table_data in tables:
                if not table_data or len(table_data) < 2:
                    continue

                headers = [str(c).strip() if c else "" for c in table_data[0]]
                rows    = [
                    [str(c).strip() if c else "" for c in row]
                    for row in table_data[1:]
                    if any(c for c in row)
                ]

                if not headers or not rows:
                    continue

                breadcrumb = f"BharatConnect > {doc_name} > {current_section}: "

                if _is_catalogue_table(headers):
                    # Rule 2 — one chunk per row
                    _flush_prose()
                    for row in rows:
                        content = _serialise_catalogue_row(headers, row, breadcrumb)
                        if content.strip().rstrip("."):
                            # Determine finer topic for plan rows
                            row_topic = topic
                            row_lower = content.lower()
                            if any(k in row_lower for k in ("pp-", "prepaid", "daily")):
                                row_topic = "plans_prepaid"
                            elif any(k in row_lower for k in ("post-", "postpaid", "rental")):
                                row_topic = "plans_postpaid"

                            chunks.append(Chunk(
                                content    = content,
                                doc_id     = doc_id,
                                source     = source,
                                section    = current_section,
                                chunk_type = "table_row",
                                topic      = row_topic,
                                priority   = "normal",
                                chunk_index = chunk_idx,
                            ))
                            chunk_idx += 1

                elif _is_comparison_table(headers):
                    # Rule 3 — whole table as one chunk
                    _flush_prose()
                    caption  = f"{current_section} — comparison table"
                    content  = _serialise_full_table(headers, rows, caption, breadcrumb)
                    chunks.append(Chunk(
                        content    = content,
                        doc_id     = doc_id,
                        source     = source,
                        section    = current_section,
                        chunk_type = "table_full",
                        topic      = topic,
                        priority   = "normal",
                        chunk_index = chunk_idx,
                    ))
                    chunk_idx += 1

                else:
                    # Generic table — serialise as full table
                    _flush_prose()
                    caption = f"{current_section} — reference table"
                    content = _serialise_full_table(headers, rows, caption, breadcrumb)
                    chunks.append(Chunk(
                        content    = content,
                        doc_id     = doc_id,
                        source     = source,
                        section    = current_section,
                        chunk_type = "table_full",
                        topic      = topic,
                        priority   = "normal",
                        chunk_index = chunk_idx,
                    ))
                    chunk_idx += 1

            # ── Process text lines on this page ────────────────────────────
            page_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

            for line in page_text.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Detect callout trigger lines
                if _is_callout_line(line):
                    _flush_prose()
                    pending_callout += line + " "
                    continue

                # If we were accumulating a callout, keep collecting until blank
                if pending_callout:
                    if line:
                        pending_callout += line + " "
                    else:
                        _flush_callout()
                    continue

                # Detect heading — start new section
                if _HEADING_RE.match(line) and len(line) < 120:
                    _flush_prose()
                    current_section = line
                    continue

                # Otherwise: regular prose
                current_prose += line + " "

        # Flush any remaining content after last page
        _flush_callout()
        _flush_prose()

    logger.info(
        f"Extracted from '{source}': {len(chunks)} chunks  "
        f"(prose={sum(1 for c in chunks if c.chunk_type=='prose')}, "
        f"table_row={sum(1 for c in chunks if c.chunk_type=='table_row')}, "
        f"table_full={sum(1 for c in chunks if c.chunk_type=='table_full')}, "
        f"callout={sum(1 for c in chunks if c.chunk_type=='callout')})"
    )
    return chunks


# ---------------------------------------------------------------------------
# Embedder + upserter
# ---------------------------------------------------------------------------

def _chunk_point_id(chunk: Chunk) -> str:
    """Deterministic Qdrant point ID from content hash."""
    digest = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
    return str(uuid.UUID(digest[:32]))


async def upsert_chunks(chunks: list[Chunk], store: QdrantStore) -> int:
    """Embed all chunks and upsert into Qdrant knowledge_base in batches."""
    total = 0
    batch: list[dict] = []

    for chunk in chunks:
        if not chunk.content.strip():
            continue

        # Use "passage" prefix for index-time embedding (E5 convention)
        vector = await embed_text(chunk.content, prefix="passage")

        point = {
            "id":     _chunk_point_id(chunk),
            "vector": vector,
            "payload": {
                "doc_id":      chunk.doc_id,
                "source":      chunk.source,
                "section":     chunk.section,
                "chunk_type":  chunk.chunk_type,
                "topic":       chunk.topic,
                "priority":    chunk.priority,
                "language":    chunk.language,
                "content":     chunk.content,
                "chunk_index": chunk.chunk_index,
            },
        }
        batch.append(point)

        if len(batch) >= UPSERT_BATCH:
            await store.upsert(KNOWLEDGE_BASE, batch)
            total += len(batch)
            logger.debug(f"Upserted batch of {len(batch)} chunks")
            batch = []

    if batch:
        await store.upsert(KNOWLEDGE_BASE, batch)
        total += len(batch)

    return total


# ---------------------------------------------------------------------------
# Public: ingest one PDF
# ---------------------------------------------------------------------------

async def ingest_pdf(pdf_path: Path, store: QdrantStore) -> dict:
    """
    Ingest one BharatConnect PDF into Qdrant.

    Returns a summary dict: {doc_id, total, by_type}.
    """
    filename = pdf_path.name
    doc_meta = DOC_REGISTRY.get(filename)

    if doc_meta is None:
        # Unknown PDF — fall back to generic ingestion via KnowledgeBaseLoader
        logger.warning(
            f"'{filename}' not in DOC_REGISTRY — falling back to generic loader"
        )
        from app.knowledge.embedder import KnowledgeBaseLoader
        loader = KnowledgeBaseLoader(store)
        count  = await loader.load_pdf(str(pdf_path))
        return {"doc_id": filename, "total": count, "by_type": {"prose": count}}

    chunks = extract_chunks(pdf_path, doc_meta)
    total  = await upsert_chunks(chunks, store)

    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1

    logger.info(
        f"Ingested '{filename}' ({doc_meta['doc_id']}): "
        f"{total} points upserted  {by_type}"
    )
    return {"doc_id": doc_meta["doc_id"], "total": total, "by_type": by_type}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _cli_main():
    parser = argparse.ArgumentParser(
        description="Ingest BharatConnect PDFs into the DreamSupport knowledge base"
    )
    parser.add_argument(
        "--file", default=None,
        help="Path to a single PDF file. Omit to ingest all 5 BharatConnect PDFs."
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Print chunk counts per doc_id from Qdrant (no ingestion)."
    )
    args = parser.parse_args()

    store = QdrantStore(QDRANT_URL, QDRANT_API_KEY)
    await store.init_collections()

    if args.verify:
        await _verify(store)
        return

    if args.file:
        pdf_path = Path(args.file)
        if not pdf_path.exists():
            print(f"File not found: {pdf_path}")
            return
        result = await ingest_pdf(pdf_path, store)
        _print_result([result])
    else:
        # Ingest all 5 PDFs
        results = []
        for filename in DOC_REGISTRY:
            pdf_path = DATA_DIR / filename
            if not pdf_path.exists():
                logger.warning(f"PDF not found, skipping: {pdf_path}")
                continue
            result = await ingest_pdf(pdf_path, store)
            results.append(result)
        _print_result(results)

    await store.close()


async def _verify(store: QdrantStore) -> None:
    """Print chunk counts from Qdrant per doc_id."""
    print("\n── Knowledge Base Verification ──────────────────────────────")
    total_all = 0
    for filename, meta in DOC_REGISTRY.items():
        doc_id = meta["doc_id"]
        try:
            records, _ = await store.scroll(
                KNOWLEDGE_BASE,
                filter=store.filter_eq("doc_id", doc_id),
                limit=2000,
            )
            count = len(records)
            total_all += count

            by_type: dict[str, int] = {}
            for r in records:
                ct = r["payload"].get("chunk_type", "unknown")
                by_type[ct] = by_type.get(ct, 0) + 1

            type_str = "  ".join(f"{k}={v}" for k, v in sorted(by_type.items()))
            print(f"  {doc_id}  {filename[:50]:<50}  {count:>4} chunks  ({type_str})")
        except Exception as exc:
            print(f"  {doc_id}  ERROR: {exc}")

    print(f"\n  TOTAL: {total_all} chunks in knowledge_base")
    print("──────────────────────────────────────────────────────────────\n")


def _print_result(results: list[dict]) -> None:
    print("\n── Ingestion Complete ────────────────────────────────────────")
    total = 0
    for r in results:
        type_str = "  ".join(f"{k}={v}" for k, v in sorted(r["by_type"].items()))
        print(f"  {r['doc_id']}  {r['total']:>4} chunks  ({type_str})")
        total += r["total"]
    print(f"\n  TOTAL: {total} chunks upserted into Qdrant knowledge_base ✓")
    print("──────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    asyncio.run(_cli_main())
