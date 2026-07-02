"""
retriever.py — Upgraded RAG retrieval pipeline  (Phase 2 + Ingestion upgrade)
==============================================================================

RetrievalPipeline.retrieve(query, language, top_k)

  1. Critical chunk injection   — deterministic, O(1) fetch by payload filter.
     Triggered by purchase/OTP keywords. Always prepended to results.
  2. Topic pre-filter           — rule-based keyword classifier (~0ms).
     Maps query to a Qdrant payload filter (topic=billing, etc.) so the
     vector search only scans the relevant doc subset.
  3. Embed query                — using multilingual-e5-small with "query: " prefix.
  4. Vector search              — Qdrant cosine similarity, with or without filter.
  5. Optional reranker          — cross-encoder, off by default.
  6. Return merged results      — critical chunks first, then similarity results.

Multilingual support
--------------------
  Keyword sets include Hindi/Marathi/Tamil equivalents for the most common
  telecom intents.  The embedding model (multilingual-e5-small) handles the
  rest — Indic script queries map into the same vector space as English chunks.

Critical chunk guarantee
------------------------
  The "recharges only on website" rule and the "never share OTP" policy are
  safety-critical.  They are fetched deterministically by payload filter
  (doc_id + priority=critical) so they always surface on relevant queries
  regardless of vector similarity score.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from loguru import logger

from app.knowledge.embedder import embed_text
from app.store import KNOWLEDGE_BASE, QdrantStore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MIN_SCORE_THRESHOLD = float(os.getenv("RAG_MIN_SCORE", "0.25"))   # slightly lower for multilingual

# If the best topic-filtered hit scores below MIN_SCORE_THRESHOLD + this margin,
# the filter is probably too narrow (e.g. mis-classified topic) — run one extra
# UNfiltered search and merge the results, so a weak on-topic chunk can't suppress
# a much better off-topic one (Bug #9). Adds at most one extra search, well within
# the 1.5s RAG guard. Set to 0 to disable the low-confidence retry.
RAG_FILTER_CONFIDENCE_MARGIN = float(os.getenv("RAG_FILTER_CONFIDENCE_MARGIN", "0.10"))

RERANKER_ENABLED = os.getenv("RERANKER", "false").lower() == "true"
RERANKER_MODEL   = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

_reranker = None


# ---------------------------------------------------------------------------
# RetrievedChunk
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    doc_id:     str
    content:    str
    source:     str
    score:      float
    chunk_type: str  = "prose"
    topic:      str  = "general"
    priority:   str  = "normal"
    section:    str  = ""
    metadata:   dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Topic pre-filter  (rule-based, ~0 ms)
# ---------------------------------------------------------------------------

# Keywords per topic — includes Hindi/Marathi equivalents for top intents
_TOPIC_KEYWORDS: dict[str, set[str]] = {
    "billing": {
        # English
        "bill", "charge", "recharge", "plan", "pack", "pay", "payment",
        "amount", "balance", "refund", "invoice", "due", "deduct", "prepaid",
        "postpaid", "pp-", "post-", "add-on", "data pack", "expired", "validity",
        "topup", "top-up", "renewal", "auto-renew", "price", "cost", "rate",
        # Hindi
        "बिल", "रिचार्ज", "प्लान", "पैसे", "पेमेंट", "बैलेंस", "रिफंड",
        "चार्ज", "डेटा", "वैधता",
        # Marathi
        "बिल", "रिचार्ज", "योजना", "पैसे", "शिल्लक",
    },
    "network": {
        # English
        "network", "signal", "5g", "4g", "3g", "slow", "internet", "speed",
        "coverage", "connectivity", "fiber", "fibre", "broadband", "wifi",
        "wi-fi", "outage", "down", "not working", "weak signal", "no signal",
        "tower", "roaming", "international",
        # Hindi
        "नेटवर्क", "सिग्नल", "स्पीड", "इंटरनेट", "कवरेज",
        # Marathi
        "नेटवर्क", "जाळे", "वेग",
    },
    "policy": {
        # English
        "otp", "fraud", "kyc", "complaint", "grievance", "port", "mnp",
        "portability", "cancel", "cancellation", "terminate", "disconnect",
        "privacy", "data protection", "identity", "documents", "verification",
        "escalate", "escalation", "docket", "trai", "legal", "terms",
        "refund policy", "sim", "lost sim", "block sim",
        # Hindi
        "शिकायत", "धोखाधड़ी", "पोर्ट",
    },
    "competitive": {
        # English
        "competitor", "jio", "airtel", "vi", "vodafone", "idea", "bsnl",
        "better", "switch", "compare", "comparison", "cheaper", "best plan",
        "other network", "telanova", "speedcell", "vistatel",
        # Hindi
        "प्रतियोगी", "दूसरा नेटवर्क",
    },
}


# Plan/price catalogue rows are ingested under FINER topics than "billing"
# (ingestor.py: per-row plan chunks get topic="plans_prepaid"/"plans_postpaid",
# while general billing prose stays "billing"). A "billing" filter alone
# therefore EXCLUDES the exact price rows a plan/price query needs. Whenever we
# select the billing topic we must widen the filter to include these siblings.
_BILLING_TOPIC_GROUP = ["billing", "plans_prepaid", "plans_postpaid"]


def _detect_topic(query: str) -> list[str] | None:
    """
    Map a query to the set of Qdrant topic values to filter on (OR semantics).

    Returns a list of topics if confident, None if ambiguous.
    Ambiguous queries (match 0 or ≥2 unrelated topics) return None → full search.
    When "billing" is selected, the returned list also includes the split
    plan-catalogue topics so plan/price rows are not filtered out.
    """
    q_lower = query.lower()
    matched: list[str] = []

    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            matched.append(topic)

    def _expand(topic: str) -> list[str]:
        # Widen billing to include the split plan-catalogue topics.
        return list(_BILLING_TOPIC_GROUP) if topic == "billing" else [topic]

    if len(matched) == 1:
        return _expand(matched[0])
    # billing+policy overlap is common (refund questions) → use billing group
    if set(matched) == {"billing", "policy"}:
        return list(_BILLING_TOPIC_GROUP)
    return None


# ---------------------------------------------------------------------------
# Critical chunk triggers
# ---------------------------------------------------------------------------

_PURCHASE_TRIGGERS = {
    # English
    "recharge", "buy", "purchase", "pay for", "get a plan", "add-on",
    "how to pay", "where to pay", "website", "online payment",
    # Hindi
    "रिचार्ज", "खरीद", "खरीदना", "भुगतान",
    # Marathi
    "रिचार्ज", "विकत", "पैसे भर",
}

_OTP_TRIGGERS = {
    # English
    "otp", "pin", "password", "cvv", "card number", "bank details",
    "account number", "verification code", "security code",
    # Hindi
    "ओटीपी", "पिन", "पासवर्ड",
    # Marathi
    "ओटीपी", "संकेतांक",
}


def _needs_critical_chunks(query: str) -> tuple[bool, bool]:
    """Returns (inject_purchase_rule, inject_otp_rule)."""
    q_lower = query.lower()
    purchase = any(t in q_lower for t in _PURCHASE_TRIGGERS)
    otp      = any(t in q_lower for t in _OTP_TRIGGERS)
    return purchase, otp


# ---------------------------------------------------------------------------
# RetrievalPipeline
# ---------------------------------------------------------------------------

class RetrievalPipeline:
    """
    Full retrieval pipeline with topic pre-filtering and critical chunk injection.
    One instance per app lifetime — passed in at startup from main.py lifespan.
    """

    def __init__(self, store: QdrantStore) -> None:
        self._store = store

    async def retrieve(
        self,
        query:    str,
        language: str = "en-IN",
        top_k:    int = 5,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant knowledge-base chunks for a query.

        Args:
            query:    raw user query (any language)
            language: BCP-47 code of detected language (for logging)
            top_k:    max normal results to return (critical chunks are extra)

        Returns an empty list when the knowledge base has no content yet.
        """
        if not query or not query.strip():
            return []

        # 1. Critical chunk injection (deterministic, always runs)
        inject_purchase, inject_otp = _needs_critical_chunks(query)
        critical_chunks: list[RetrievedChunk] = []

        if inject_purchase:
            cc = await self._fetch_critical("KB-CORP-001")
            critical_chunks.extend(cc)
            logger.debug(f"retrieve: injected {len(cc)} purchase-rule critical chunks")

        if inject_otp:
            cc = await self._fetch_critical("KB-POL-002")
            critical_chunks.extend(cc)
            logger.debug(f"retrieve: injected {len(cc)} OTP-guard critical chunks")

        # 2. Topic pre-filter (OR over a set of topics; see _detect_topic)
        topics = _detect_topic(query)
        filt   = self._store.filter_in("topic", topics) if topics else None
        if topics:
            logger.debug(f"retrieve: topic filter → topics={topics}")

        # 3. Embed query with "query: " prefix (E5 convention)
        query_vector = await embed_text(query, prefix="query")

        # 4. Vector search
        fetch_k = top_k * 3 if RERANKER_ENABLED else top_k
        try:
            raw_results = await self._store.search(
                collection = KNOWLEDGE_BASE,
                vector     = query_vector,
                filter     = filt,
                limit      = fetch_k,
            )
        except Exception as exc:
            logger.error(f"RetrievalPipeline.retrieve: Qdrant search failed: {exc}")
            # Fallback: return only critical chunks if we have them
            return critical_chunks

        # Fallback / low-confidence rescue for topic-filtered searches.
        # (a) Zero results — the classic empty case.
        # (b) Best hit is weak (top score < MIN + margin) — the filter is likely
        #     too narrow (e.g. mis-classified topic), so a much better off-topic
        #     chunk is being suppressed. In both cases run ONE unfiltered search
        #     and MERGE, so we never lose the filtered hits but can surface better
        #     unfiltered ones. (Bug #9)
        if filt is not None:
            top_score      = raw_results[0]["score"] if raw_results else 0.0
            low_confidence = top_score < (MIN_SCORE_THRESHOLD + RAG_FILTER_CONFIDENCE_MARGIN)
            if not raw_results or (RAG_FILTER_CONFIDENCE_MARGIN > 0 and low_confidence):
                reason = "0 results" if not raw_results else f"low top score {top_score:.3f}"
                logger.debug(f"retrieve: topic filter weak ({reason}) — merging an unfiltered search")
                try:
                    unfiltered = await self._store.search(
                        collection = KNOWLEDGE_BASE,
                        vector     = query_vector,
                        limit      = fetch_k,
                    )
                except Exception as exc:
                    logger.error(f"RetrievalPipeline.retrieve: fallback search failed: {exc}")
                    if not raw_results:
                        return critical_chunks
                    unfiltered = []
                # Merge: keep filtered hits + any new unfiltered hits, dedup by
                # point id, then sort by score so the best chunks win downstream.
                seen_ids = {h["id"] for h in raw_results}
                raw_results = raw_results + [h for h in unfiltered if h["id"] not in seen_ids]
                raw_results.sort(key=lambda h: h["score"], reverse=True)

        # 5. Build result list, filter by min score, deduplicate vs critical chunks
        critical_ids = {c.doc_id + c.content[:50] for c in critical_chunks}
        chunks: list[RetrievedChunk] = []

        for hit in raw_results:
            score   = hit["score"]
            payload = hit["payload"]
            if score < MIN_SCORE_THRESHOLD:
                continue
            dedup_key = payload.get("doc_id", "") + (payload.get("content") or "")[:50]
            if dedup_key in critical_ids:
                continue   # already in critical_chunks, skip duplicate

            chunks.append(RetrievedChunk(
                doc_id     = payload.get("doc_id",     str(hit["id"])),
                content    = payload.get("content",    ""),
                source     = payload.get("source",     "unknown"),
                score      = round(score, 4),
                chunk_type = payload.get("chunk_type", "prose"),
                topic      = payload.get("topic",      "general"),
                priority   = payload.get("priority",   "normal"),
                section    = payload.get("section",    ""),
                metadata   = {
                    "chunk_type": payload.get("chunk_type", ""),
                    "section":    payload.get("section", ""),
                },
            ))

        # 6. Optional reranker
        if RERANKER_ENABLED and len(chunks) > top_k:
            chunks = await _rerank(query, chunks, top_k)
        else:
            chunks = chunks[:top_k]

        # 7. Merge: critical chunks first, then similarity results
        result = critical_chunks + chunks

        if result:
            top_score = f"{chunks[0].score:.3f}" if chunks else "n/a"
            logger.info(
                f"retrieve({query[:60]!r}, lang={language}): "
                f"{len(critical_chunks)} critical + {len(chunks)} similarity chunks "
                f"[top_score={top_score}]"
            )
        else:
            logger.debug(f"retrieve({query[:60]!r}): no chunks above threshold")

        return result

    async def _fetch_critical(self, doc_id: str) -> list[RetrievedChunk]:
        """
        Fetch critical callout chunks for a specific doc_id by payload filter.
        O(1) — no vector computation needed.
        """
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
            filt = Filter(must=[
                FieldCondition(key="doc_id",   match=MatchValue(value=doc_id)),
                FieldCondition(key="priority", match=MatchValue(value="critical")),
            ])
            records, _ = await self._store.scroll(
                KNOWLEDGE_BASE,
                filter=filt,
                limit=3,
            )
            results = []
            seen: set[str] = set()
            for r in records:
                content = r["payload"].get("content", "")
                # Deduplicate the redundant callout copies
                key = content[:80]
                if key in seen:
                    continue
                seen.add(key)
                results.append(RetrievedChunk(
                    doc_id     = r["payload"].get("doc_id", doc_id),
                    content    = content,
                    source     = r["payload"].get("source", ""),
                    score      = 1.0,   # deterministically injected = perfect relevance
                    chunk_type = "callout",
                    topic      = r["payload"].get("topic", "policy"),
                    priority   = "critical",
                    section    = r["payload"].get("section", ""),
                ))
            return results
        except Exception as exc:
            logger.warning(f"_fetch_critical({doc_id}): failed (non-fatal): {exc}")
            return []


# ---------------------------------------------------------------------------
# Optional cross-encoder reranker
# ---------------------------------------------------------------------------

async def _rerank(
    query:      str,
    candidates: list[RetrievedChunk],
    top_k:      int,
) -> list[RetrievedChunk]:
    import asyncio

    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading cross-encoder reranker '{RERANKER_MODEL}' …")
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("Reranker loaded ✓")

    pairs  = [(query, c.content) for c in candidates]
    loop   = asyncio.get_event_loop()
    scores = await loop.run_in_executor(
        None,
        lambda: _reranker.predict(pairs).tolist(),
    )

    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    result   = []
    for chunk, score in reranked[:top_k]:
        chunk.score = round(float(score), 4)
        result.append(chunk)

    logger.info(f"Reranker: {len(candidates)} → {len(result)} chunks")
    return result
