"""
store.py — Qdrant Cloud persistence layer (Phase 1)
====================================================

Single entry point for ALL vector database operations.  Five collections:

  knowledge_base      — RAG document chunks (real 768-d embeddings)
  execution_traces    — per-turn LLM traces (dummy vector, payload-only)
  customer_profiles   — per-customer memory (dummy vector, payload-only)
  dream_checkpoints   — Dream Engine progress (dummy vector, payload-only)
  improvement_log     — Dream Cycle outputs  (real embeddings of improvement text)

"Dummy vector" collections use [0.0]*768 with Distance.DOT and are never
queried by vector similarity — only by payload filters via scroll().
This avoids embedding cost while still using a single Qdrant backend.

Thread-safety: QdrantAsyncClient is async and internally manages a connection
pool, so a single QdrantStore instance is safe to share across all coroutines.
"""

from __future__ import annotations

import uuid
from typing import Any

from loguru import logger
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.config import EMBEDDING_DIM, QDRANT_API_KEY, QDRANT_URL

# ---------------------------------------------------------------------------
# Collection names (single source of truth)
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE      = "knowledge_base"
EXECUTION_TRACES    = "execution_traces"
CUSTOMER_PROFILES   = "customer_profiles"
DREAM_CHECKPOINTS   = "dream_checkpoints"
IMPROVEMENT_LOG     = "improvement_log"

_ALL_COLLECTIONS = [
    KNOWLEDGE_BASE,
    EXECUTION_TRACES,
    CUSTOMER_PROFILES,
    DREAM_CHECKPOINTS,
    IMPROVEMENT_LOG,
]

# Dummy vector used for payload-only collections (no semantic search needed).
_DUMMY_VECTOR: list[float] = [0.0] * EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Collection schemas
# ---------------------------------------------------------------------------

def _collection_config(use_dummy: bool) -> VectorParams:
    """
    Real collections use Cosine distance (meaningful for sentence-transformer
    embeddings).  Dummy collections use Dot product — all vectors are zero so
    the distance is always 0 and we never sort by it.
    """
    return VectorParams(
        size=EMBEDDING_DIM,
        distance=Distance.COSINE if not use_dummy else Distance.DOT,
    )


# Payload indexes improve scroll() / filter() performance.
# (key, schema_type) pairs per collection.
_PAYLOAD_INDEXES: dict[str, list[tuple[str, PayloadSchemaType]]] = {
    KNOWLEDGE_BASE: [
        ("doc_id",     PayloadSchemaType.KEYWORD),
        ("source",     PayloadSchemaType.KEYWORD),
        ("chunk_type", PayloadSchemaType.KEYWORD),  # prose|table_row|table_full|callout
        ("topic",      PayloadSchemaType.KEYWORD),  # billing|policy|network|plans_prepaid|plans_postpaid|competitive
        ("priority",   PayloadSchemaType.KEYWORD),  # critical|normal
    ],
    EXECUTION_TRACES: [
        ("session_id",       PayloadSchemaType.KEYWORD),
        ("dream_processed",  PayloadSchemaType.BOOL),
        ("created_at",       PayloadSchemaType.KEYWORD),
    ],
    CUSTOMER_PROFILES: [
        ("customer_id",      PayloadSchemaType.KEYWORD),
    ],
    DREAM_CHECKPOINTS: [
        ("cycle_type",       PayloadSchemaType.KEYWORD),
        ("status",           PayloadSchemaType.KEYWORD),
    ],
    IMPROVEMENT_LOG: [
        ("category",         PayloadSchemaType.KEYWORD),
        ("approved",         PayloadSchemaType.BOOL),
        ("applied_at",       PayloadSchemaType.KEYWORD),
    ],
}


# ---------------------------------------------------------------------------
# QdrantStore
# ---------------------------------------------------------------------------

class QdrantStore:
    """
    Async Qdrant wrapper.  Instantiate once at app startup and pass around.

    Usage:
        store = QdrantStore(QDRANT_URL, QDRANT_API_KEY)
        await store.init_collections()
    """

    def __init__(self, url: str = "", api_key: str = "") -> None:
        _url     = url     or QDRANT_URL
        _api_key = api_key or QDRANT_API_KEY

        if not _url:
            raise ValueError(
                "QDRANT_URL is not set. "
                "Add it to .env: QDRANT_URL=https://xxxx.qdrant.io"
            )
        if not _api_key:
            raise ValueError(
                "QDRANT_API_KEY is not set. "
                "Add it to .env: QDRANT_API_KEY=your_key_here"
            )

        self._client = AsyncQdrantClient(url=_url, api_key=_api_key)
        logger.info(f"QdrantStore initialised → {_url}")

    # ------------------------------------------------------------------
    # Collection bootstrap
    # ------------------------------------------------------------------

    async def init_collections(self) -> None:
        """
        Idempotently create all 5 collections and their payload indexes.
        Safe to call on every startup — existing collections are left intact.
        """
        existing = {c.name for c in (await self._client.get_collections()).collections}
        logger.info(f"Existing Qdrant collections: {existing or '(none)'}")

        dummy_collections = {
            EXECUTION_TRACES,
            CUSTOMER_PROFILES,
            DREAM_CHECKPOINTS,
        }

        for name in _ALL_COLLECTIONS:
            if name in existing:
                logger.debug(f"Collection '{name}' already exists — skipping")
                continue

            use_dummy = name in dummy_collections
            await self._client.create_collection(
                collection_name=name,
                vectors_config=_collection_config(use_dummy),
            )
            logger.info(f"Created collection '{name}' (dummy={use_dummy})")

        await self._ensure_payload_indexes()
        logger.info("All Qdrant collections ready ✓")

    async def _ensure_payload_indexes(self) -> None:
        """Create payload indexes for efficient filter queries."""
        for collection, fields in _PAYLOAD_INDEXES.items():
            for field_name, schema_type in fields:
                try:
                    await self._client.create_payload_index(
                        collection_name=collection,
                        field_name=field_name,
                        field_schema=schema_type,
                    )
                    logger.debug(
                        f"Payload index '{field_name}' on '{collection}' ensured"
                    )
                except Exception as exc:
                    # Index already exists — Qdrant raises an error for re-creation.
                    if "already exists" in str(exc).lower():
                        logger.debug(
                            f"Payload index '{field_name}' on '{collection}' already exists"
                        )
                    else:
                        logger.warning(
                            f"Could not create index '{field_name}' on '{collection}': {exc}"
                        )

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    async def upsert(
        self,
        collection: str,
        points: list[dict[str, Any]],
    ) -> None:
        """
        Upsert one or more points.

        Each dict must have:
          - "id"      : str | int  (UUID string recommended)
          - "vector"  : list[float]  (use _DUMMY_VECTOR for payload-only)
          - "payload" : dict[str, Any]

        Example:
            await store.upsert(EXECUTION_TRACES, [{
                "id": str(uuid.uuid4()),
                "vector": store.dummy_vector(),
                "payload": {"session_id": "...", "user_input": "..."},
            }])
        """
        structs = [
            PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p.get("payload", {}),
            )
            for p in points
        ]
        await self._client.upsert(collection_name=collection, points=structs)
        logger.debug(f"Upserted {len(structs)} point(s) into '{collection}'")

    async def search(
        self,
        collection: str,
        vector: list[float],
        filter: Filter | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Vector similarity search with optional payload filter.

        Returns a list of dicts:
            [{"id": ..., "score": float, "payload": {...}}, ...]
        """
        from qdrant_client.models import QueryRequest
        result = await self._client.query_points(
            collection_name=collection,
            query=vector,
            query_filter=filter,
            limit=limit,
            with_payload=True,
        )
        return [
            {"id": hit.id, "score": hit.score, "payload": hit.payload or {}}
            for hit in result.points
        ]

    async def scroll(
        self,
        collection: str,
        filter: Filter | None = None,
        limit: int = 100,
        offset: str | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """
        Payload-only query (no vector ranking) by default.  Used for
        non-semantic lookups such as fetching unprocessed traces or customer
        profiles. Pass with_vectors=True when the caller needs to re-rank
        results itself (e.g. an in-process cache doing local cosine scoring)
        without paying a separate search() round-trip per lookup.

        Returns (records, next_page_offset).
        Pass next_page_offset back as `offset` to paginate.

        Each record: {"id": ..., "payload": {...}} — plus "vector": [...] when
        with_vectors=True.
        """
        records, next_offset = await self._client.scroll(
            collection_name=collection,
            scroll_filter=filter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )
        items = [
            {
                "id": r.id,
                "payload": r.payload or {},
                **({"vector": r.vector} if with_vectors else {}),
            }
            for r in records
        ]
        return items, next_offset

    async def update_payload(
        self,
        collection: str,
        point_id: str,
        payload_patch: dict[str, Any],
    ) -> None:
        """
        Partial payload update — merges payload_patch into the existing payload.
        Does NOT overwrite fields that are not in payload_patch.
        """
        await self._client.set_payload(
            collection_name=collection,
            payload=payload_patch,
            points=[point_id],
        )
        logger.debug(
            f"Updated payload on '{collection}' point {point_id}: "
            f"{list(payload_patch.keys())}"
        )

    async def delete_points(
        self,
        collection: str,
        point_ids: list[str],
    ) -> None:
        """Delete points by ID list."""
        await self._client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=point_ids),
        )
        logger.debug(f"Deleted {len(point_ids)} point(s) from '{collection}'")

    async def get_point(
        self,
        collection: str,
        point_id: str,
    ) -> dict[str, Any] | None:
        """
        Fetch a single point by ID.  Returns None if not found.
        """
        results = await self._client.retrieve(
            collection_name=collection,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return None
        r = results[0]
        return {"id": r.id, "payload": r.payload or {}}

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @staticmethod
    def dummy_vector() -> list[float]:
        """Return the zero-vector used for payload-only collections."""
        return _DUMMY_VECTOR.copy()

    @staticmethod
    def new_id() -> str:
        """Generate a new UUID string suitable as a Qdrant point ID."""
        return str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Convenience filter builders (avoid importing qdrant_client everywhere)
    # ------------------------------------------------------------------

    @staticmethod
    def filter_eq(field: str, value: Any) -> Filter:
        """Build a simple equality filter: field == value."""
        return Filter(
            must=[FieldCondition(key=field, match=MatchValue(value=value))]
        )

    @staticmethod
    def filter_in(field: str, values: list[Any]) -> Filter:
        """Build a set-membership filter: field IN values (OR)."""
        return Filter(
            must=[FieldCondition(key=field, match=MatchAny(any=list(values)))]
        )

    @staticmethod
    def filter_and(*conditions: Filter) -> Filter:
        """Combine multiple filters with AND logic."""
        must_clauses = []
        for f in conditions:
            if f.must:
                must_clauses.extend(f.must)
        return Filter(must=must_clauses)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        """
        Returns cluster info and per-collection point counts.
        Used by the /health endpoint to confirm Qdrant connectivity.
        """
        info = await self._client.get_collections()
        counts: dict[str, int] = {}
        for col in info.collections:
            col_info = await self._client.get_collection(col.name)
            counts[col.name] = col_info.points_count or 0
        return {"status": "ok", "collections": counts}

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP client.  Call from app lifespan teardown."""
        await self._client.close()
        logger.info("QdrantStore closed")
