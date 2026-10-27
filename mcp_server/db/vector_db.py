"""
Vector database module (Qdrant) for semantic search over bond document
content - primarily the "prose" chunks of a canonical BondDocument
(Risk Factors, Objects of the Issue, Definitions, procedural text),
plus "structured_fact" chunks for when semantic search is a better
entry point than an exact SQL filter.

Design notes:
- Embedding generation is intentionally pluggable (`embed_fn`) so this
  module doesn't hard-code an embedding provider. Wire it to whatever
  the ingestion pipeline used (must match embedding_dim in config).
- Payload schema matches src/schema/canonical_schema.py's Chunk model
  directly (see scripts/load_canonical_to_qdrant.py), with issuer/
  document_type added at load time since Chunk doesn't carry them
  (they're document-level metadata, not per-chunk):

    {
      "chunk_id": str,          # e.g. "pfc_ncd_2026_chunk_series_III_cat_III"
      "chunk_type": str,        # "structured_fact" | "prose"
      "text": str,
      "document_name": str,     # e.g. "pfc_ncd_2026.pdf"
      "issuer": str,            # added at ingestion time, not from Chunk
      "document_type": str,     # added at ingestion time, not from Chunk
      "section": str,           # top-level heading, prose chunks only
      "subsection": str,
      "section_path": str,      # full hierarchy joined with " > "
      "series_id": str | None,  # set on structured_fact chunks
      "category_id": str | None,
      "start_page": int,
      "end_page": int,
    }

The original version of this payload schema (written before
canonical_schema.py's merged Chunk model existed) matched only the
standalone chunker.py's raw output - it had no chunk_type, series_id,
or category_id at all, because "structured_fact" chunks didn't exist
yet in that pipeline version. Without chunk_type, vector_search could
never tell a clean structured fact from raw prose, and without
series_id/category_id it couldn't be scoped to a specific
series/category before falling back to a corpus-wide semantic search.
"""

import logging
from typing import Any, Callable

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import ApiException

from config import qdrant_config, server_config
from models import Citation, ToolError, ToolSuccess

logger = logging.getLogger("mcp.vector")

EmbedFn = Callable[[str], list[float]]


class VectorDatabase:
    def __init__(self, embed_fn: EmbedFn | None = None) -> None:
        self._client = QdrantClient(
            host=qdrant_config.host,
            port=qdrant_config.port,
            api_key=qdrant_config.api_key or None,
            timeout=server_config.query_timeout_seconds,
        )
        # Injected at startup in server.py once an embedding provider is chosen.
        self._embed_fn = embed_fn

    def set_embed_fn(self, embed_fn: EmbedFn) -> None:
        self._embed_fn = embed_fn

    def search(
        self,
        query_text: str,
        top_k: int = 8,
        issuer_filter: str | None = None,
        document_type_filter: str | None = None,
        chunk_type_filter: str | None = None,
        series_id_filter: str | None = None,
        category_id_filter: str | None = None,
        score_threshold: float | None = None,
    ) -> ToolSuccess | ToolError:
        if self._embed_fn is None:
            return ToolError(error="embedding_provider_not_configured",
                              detail="Call set_embed_fn() during server startup before using vector_search.")

        try:
            query_vector = self._embed_fn(query_text)
        except Exception as exc:  # embedding provider failure
            logger.exception("Embedding generation failed")
            return ToolError(error="embedding_failed", detail=str(exc))

        must_conditions = []
        if issuer_filter:
            must_conditions.append(qmodels.FieldCondition(key="issuer", match=qmodels.MatchValue(value=issuer_filter)))
        if document_type_filter:
            must_conditions.append(qmodels.FieldCondition(key="document_type", match=qmodels.MatchValue(value=document_type_filter)))
        if chunk_type_filter:
            must_conditions.append(qmodels.FieldCondition(key="chunk_type", match=qmodels.MatchValue(value=chunk_type_filter)))
        if series_id_filter:
            must_conditions.append(qmodels.FieldCondition(key="series_id", match=qmodels.MatchValue(value=series_id_filter)))
        if category_id_filter:
            must_conditions.append(qmodels.FieldCondition(key="category_id", match=qmodels.MatchValue(value=category_id_filter)))

        query_filter = qmodels.Filter(must=must_conditions) if must_conditions else None

        try:
            result = self._client.query_points(
                collection_name=qdrant_config.collection,
                query=query_vector,
                query_filter=query_filter,
                limit=min(top_k, server_config.max_rows_returned),
                score_threshold=score_threshold,
                with_payload=True,
            )

            hits = result.points
        except ApiException as exc:
            logger.exception("Qdrant search failed")
            return ToolError(error="vector_search_failed", detail=str(exc))

        data: list[dict[str, Any]] = []
        citations: list[Citation] = []

        for hit in hits:
            payload = hit.payload or {}
            data.append(
                {
                    "chunk_id": payload.get("chunk_id", str(hit.id)),
                    "chunk_type": payload.get("chunk_type"),
                    "score": hit.score,
                    "text": payload.get("text", ""),
                    "document_name": payload.get("document_name"),
                    "issuer": payload.get("issuer"),
                    "document_type": payload.get("document_type"),
                    "section": payload.get("section"),
                    "subsection": payload.get("subsection"),
                    "section_path": payload.get("section_path"),
                    "series_id": payload.get("series_id"),
                    "category_id": payload.get("category_id"),
                    "start_page": payload.get("start_page"),
                    "end_page": payload.get("end_page"),
                }
            )
            citations.append(
                Citation(
                    source_type="vector",
                    source_id=payload.get("chunk_id", str(hit.id)),
                    document_name=payload.get("document_name"),
                    page_number=payload.get("start_page"),
                    section=payload.get("section_path") or payload.get("section"),
                    confidence=hit.score,
                )
            )

        return ToolSuccess(
            tool="vector_search",
            data=data,
            citations=citations,
            row_count=len(data),
            truncated=False,
        )

    def health_check(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False


vector_db = VectorDatabase()
