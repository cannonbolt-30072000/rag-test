"""
Embedder for the RAG pipeline.
Converts text chunks into dense + sparse vectors and stores in Qdrant.
"""

import uuid
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
)

# Import shared resources - no model loading here
from retrieval.db import client, dense_model, sparse_model, COLLECTION_NAME


def create_collection():
    """Create Qdrant collection with both dense and sparse vector configs."""

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=384,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )


def embed_and_store(chunks: list[dict]) -> int:
    """
    Embed chunks with both dense and sparse models, store in Qdrant.

    Args:
        chunks: Output from chunker.py - list of {"text", "metadata"} dicts.

    Returns:
        Number of points stored.
    """

    texts = [chunk["text"] for chunk in chunks]

    # Dense embeddings: semantic meaning (one batch call)
    dense_embeddings = dense_model.encode(texts, show_progress_bar=True)

    # Sparse embeddings: keyword/term matching (returns generator, so list() it)
    sparse_embeddings = list(sparse_model.embed(texts))

    # Build points with BOTH vector types
    points = []
    for chunk, dense_emb, sparse_emb in zip(chunks, dense_embeddings, sparse_embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_emb.tolist(),
                    "sparse": SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist(),
                    ),
                },
                payload={
                    "text": chunk["text"],
                    "source": chunk["metadata"]["source"],
                    "page": chunk["metadata"]["page"],
                    "chunk_index": chunk["metadata"]["chunk_index"],
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)