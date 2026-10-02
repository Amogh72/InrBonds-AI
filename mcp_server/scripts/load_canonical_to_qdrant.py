"""
Loads a canonical BondDocument JSON's `chunks` list (the real pipeline
output - see src/schema/canonical_schema.py's Chunk model) into
Qdrant, in the payload shape db/vector_db.py's search() expects.

This replaces the original load_chunks_to_qdrant.py, which read a
standalone chunker.py export (document/total_chunks/chunking_config/
chunks) that predates chunk_type, series_id, and category_id, and
predates "structured_fact" chunks existing at all - at the time that
loader was written, canonical_extractor.py's merge of structured_fact
+ prose chunks into one list didn't exist yet, so it only ever loaded
prose.

--------------------------------------------------------------------------
CONFIRMED INPUT CONTRACT (from data/processed/pfc_ncd_2026.json):

    {
      "document": {"document_id": "pfc_ncd_2026", "document_name": "pfc_ncd_2026.pdf",
                    "document_type": "Bond/NCD Prospectus", ...},
      "issuer": {"name": "Power Finance Corporation Limited", ...},
      "chunks": [
        {
          "chunk_id": "pfc_ncd_2026_chunk_series_I_cat_III",
          "chunk_type": "structured_fact",
          "text": "...",
          "series_id": "pfc_ncd_2026_series_I",
          "category_id": "III",
          "section_path": null,
          "start_page": 78, "end_page": 78,
          "provenance": [...]
        },
        {
          "chunk_id": "pfc_ncd_2026_0001",
          "chunk_type": "prose",
          "text": "...",
          "series_id": null, "category_id": null,
          "section_path": ["SECTION I – GENERAL", "DEFINITIONS AND ABBREVIATIONS"],
          "start_page": 3, "end_page": 3
        },
        ...
      ]
    }

issuer name and document_type are read directly from the canonical
JSON's document/issuer metadata - unlike the original loader, they
don't need to be passed in by hand (they can still be overridden via
--issuer / --document-type if you want to relabel a document).
--------------------------------------------------------------------------

Usage:
    python scripts/load_canonical_to_qdrant.py data/processed/pfc_ncd_2026.json
"""

import sys
import os
import json
import hashlib
import logging
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import qdrant_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("load_canonical_chunks")

BATCH_SIZE = 128


def _normalize_chunk(chunk: dict, document_name: str, issuer: str, document_type: str) -> dict:
    """Map one canonical Chunk record to the Qdrant payload shape."""
    section_path_list = chunk.get("section_path") or []
    section_path = " > ".join(section_path_list)

    return {
        "chunk_id": chunk["chunk_id"],
        "chunk_type": chunk["chunk_type"],
        "text": chunk["text"],
        "document_name": document_name,
        "issuer": issuer,
        "document_type": document_type,
        "section": section_path_list[0] if section_path_list else None,
        "subsection": section_path_list[-1] if section_path_list else None,
        "section_path": section_path,
        "series_id": chunk.get("series_id"),
        "category_id": chunk.get("category_id"),
        "start_page": chunk.get("start_page"),
        "end_page": chunk.get("end_page"),
    }


def _embed(text: str) -> list[float]:
    """Use the SAME embedding function wired into server.py's _default_embed_fn
    so ingestion-time and query-time embeddings never drift apart."""
    from server import _default_embed_fn
    return _default_embed_fn(text)


def _point_id(chunk_id: str) -> int:
    """Qdrant point IDs must be int or UUID - hash the string chunk_id deterministically
    so re-running this script on the same file upserts (overwrites) rather than duplicates."""
    return int(hashlib.sha256(chunk_id.encode()).hexdigest()[:16], 16)


def load_canonical_chunks(json_path: str, issuer: str | None, document_type: str | None) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        payload_json = json.load(f)

    document_meta = payload_json["document"]
    document_name = document_meta["document_name"]
    resolved_issuer = issuer or payload_json["issuer"]["name"]
    resolved_document_type = document_type or document_meta.get("document_type") or "unknown"

    raw_chunks = payload_json.get("chunks", [])
    logger.info(
        "Loaded %d chunks from %s (document=%s, issuer=%s)",
        len(raw_chunks), json_path, document_name, resolved_issuer,
    )

    client = QdrantClient(
        host=qdrant_config.host,
        port=qdrant_config.port,
        api_key=qdrant_config.api_key or None,
    )

    # This chunk export supersedes any earlier one for the same document
    # (e.g. a re-run after an extraction fix) - clear old points first so
    # stale/duplicate chunks don't linger alongside the new ones.
    client.delete(
        collection_name=qdrant_config.collection,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="document_name", match=qmodels.MatchValue(value=document_name))]
            )
        ),
    )
    logger.info("Cleared existing points for document_name=%s", document_name)

    points: list[qmodels.PointStruct] = []
    skipped = 0
    structured_fact_count = 0
    prose_count = 0

    for i, raw in enumerate(raw_chunks, start=1):
        try:
            chunk = _normalize_chunk(raw, document_name, resolved_issuer, resolved_document_type)
            if not chunk["text"].strip():
                skipped += 1
                continue
            vector = _embed(chunk["text"])
        except (KeyError, NotImplementedError) as exc:
            logger.warning("Skipping chunk %s due to %s: %s", raw.get("chunk_id", "?"), type(exc).__name__, exc)
            skipped += 1
            continue

        if chunk["chunk_type"] == "structured_fact":
            structured_fact_count += 1
        elif chunk["chunk_type"] == "prose":
            prose_count += 1

        points.append(
            qmodels.PointStruct(
                id=_point_id(chunk["chunk_id"]),
                vector=vector,
                payload=chunk,
            )
        )

        if len(points) >= BATCH_SIZE:
            client.upsert(collection_name=qdrant_config.collection, points=points)
            logger.info("Upserted batch (%d/%d)", i, len(raw_chunks))
            points = []

    if points:
        client.upsert(collection_name=qdrant_config.collection, points=points)
        logger.info("Upserted final batch of %d", len(points))

    logger.info(
        "Done. Loaded %d chunks (%d structured_fact, %d prose), skipped %d.",
        len(raw_chunks) - skipped, structured_fact_count, prose_count, skipped,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a canonical BondDocument JSON's chunks into Qdrant.")
    parser.add_argument("json_path", help="Path to a canonical BondDocument JSON (e.g. data/processed/pfc_ncd_2026.json)")
    parser.add_argument("--issuer", help='Override the issuer name (defaults to document["issuer"]["name"])')
    parser.add_argument("--document-type", help='Override the document type (defaults to document["document"]["document_type"])')
    args = parser.parse_args()

    load_canonical_chunks(args.json_path, args.issuer, args.document_type)
