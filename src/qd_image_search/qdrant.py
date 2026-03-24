"""QDrant API wrapper for collection management and vector search."""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


def get_client(host: str = QDRANT_HOST, port: int = QDRANT_PORT) -> QdrantClient:
    """Create and return a QDrant client.

    Args:
        host: QDrant server hostname (default ``"localhost"``).
        port: QDrant server port (default ``6333``).

    Returns:
        A connected ``QdrantClient`` instance.
    """
    return QdrantClient(host=host, port=port)


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """Create a collection if it does not already exist.

    Args:
        client: An active ``QdrantClient``.
        collection_name: Name of the target collection.
        vector_size: Dimensionality of the embedding vectors.
    """
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def upsert_image_point(
    client: QdrantClient,
    collection_name: str,
    vector: list[float],
    metadata: dict[str, Any],
    image_b64: str,
) -> str:
    """Insert a single image point into a QDrant collection.

    Args:
        client: An active ``QdrantClient``.
        collection_name: Name of the target collection.
        vector: The CLIP embedding vector for the image.
        metadata: Dict with keys ``filename``, ``file_type``, ``width``, ``height``.
        image_b64: Base64-encoded JPEG representation of the image.

    Returns:
        The UUID string assigned to the new point.
    """
    point_id = str(uuid.uuid4())
    payload = {
        "id": point_id,
        **metadata,
        "image": image_b64,
    }
    client.upsert(
        collection_name=collection_name,
        points=[PointStruct(id=point_id, vector=vector, payload=payload)],
    )
    return point_id


def search_collection(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search a QDrant collection with a query vector.

    Args:
        client: An active ``QdrantClient``.
        collection_name: Name of the collection to search.
        query_vector: The CLIP embedding for the search query.
        limit: Maximum number of results to return (default ``10``).

    Returns:
        A list of payload dicts for the top matching points, each augmented
        with a ``score`` key.
    """
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit,
        with_payload=True,
    )
    records = []
    for hit in results:
        record = {k: v for k, v in (hit.payload or {}).items() if k != "image"}
        record["score"] = hit.score
        records.append(record)
    return records
