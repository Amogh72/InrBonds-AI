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
    client = QdrantClient(
        host=qdrant_config.host,
        port=qdrant_config.port,
        api_key=qdrant_config.api_key or None,
    )

    existing = [c.name for c in client.get_collections().collections]
    if qdrant_config.collection in existing:
        print(f"Collection '{qdrant_config.collection}' already exists. Skipping.")
        return

    client.create_collection(
        collection_name=qdrant_config.collection,
        vectors_config=qmodels.VectorParams(
            size=qdrant_config.embedding_dim,
            distance=qmodels.Distance.COSINE,
        ),
    )

    # Index payload fields used as filters in vector_db.py's search().
    # chunk_type/series_id/category_id are what let a query stay scoped
    # to e.g. "only structured_fact chunks for Series V" instead of
    # falling back to a corpus-wide semantic search (see
    # src/schema/canonical_schema.py's Chunk model).
    client.create_payload_index(qdrant_config.collection, "issuer", qmodels.PayloadSchemaType.KEYWORD)
    client.create_payload_index(qdrant_config.collection, "document_type", qmodels.PayloadSchemaType.KEYWORD)
    client.create_payload_index(qdrant_config.collection, "chunk_type", qmodels.PayloadSchemaType.KEYWORD)
    client.create_payload_index(qdrant_config.collection, "series_id", qmodels.PayloadSchemaType.KEYWORD)
    client.create_payload_index(qdrant_config.collection, "category_id", qmodels.PayloadSchemaType.KEYWORD)

    print(f"Created collection '{qdrant_config.collection}' "
          f"(dim={qdrant_config.embedding_dim}, distance=COSINE) with "
          f"issuer/document_type/chunk_type/series_id/category_id indexes.")


if __name__ == "__main__":
    main()
