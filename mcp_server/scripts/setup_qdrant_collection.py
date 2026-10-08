"""
Run once to create the Qdrant collection with the correct vector size
before your ingestion pipeline starts writing document chunks to it.

    python scripts/setup_qdrant_collection.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from config import qdrant_config


def main():
    # Generous timeout: index creation can take several seconds under slow
    # disk I/O (seen in practice on Docker Desktop for Windows), well past
    # qdrant-client's short default - without this the request can still
    # succeed server-side after the client has already given up and raised
    # ReadTimeout, leaving the script's own state out of sync with Qdrant's.
    client = QdrantClient(
        host=qdrant_config.host,
        port=qdrant_config.port,
        api_key=qdrant_config.api_key or None,
        timeout=60,
    )

    existing = [c.name for c in client.get_collections().collections]
    if qdrant_config.collection in existing:
        print(f"Collection '{qdrant_config.collection}' already exists.")
    else:
        client.create_collection(
            collection_name=qdrant_config.collection,
            vectors_config=qmodels.VectorParams(
                size=qdrant_config.embedding_dim,
                distance=qmodels.Distance.COSINE,
            ),
        )
        print(f"Created collection '{qdrant_config.collection}' "
              f"(dim={qdrant_config.embedding_dim}, distance=COSINE).")

    # Index payload fields used as filters in vector_db.py's search().
    # chunk_type/series_id/category_id are what let a query stay scoped
    # to e.g. "only structured_fact chunks for Series V" instead of
    # falling back to a corpus-wide semantic search (see
    # src/schema/canonical_schema.py's Chunk model).
    # create_payload_index is idempotent (re-running it on an already-
    # indexed field is a no-op), so always re-asserting every index here
    # - rather than skipping whenever the collection already exists - is
    # what makes this script safely re-runnable after a partial failure.
    for field in ("issuer", "document_type", "chunk_type", "series_id", "category_id"):
        client.create_payload_index(qdrant_config.collection, field, qmodels.PayloadSchemaType.KEYWORD)

    print("Payload indexes ready: issuer/document_type/chunk_type/series_id/category_id.")


if __name__ == "__main__":
    main()
