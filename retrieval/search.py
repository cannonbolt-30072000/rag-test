"""
Retrieval module for the RAG pipeline.
Hybrid search (dense + sparse) with RRF fusion and cross-encoder reranking.
"""

from qdrant_client import models

from retrieval.db import client, dense_model, sparse_model, reranker, COLLECTION_NAME


def search(query: str, top_k: int = 5, rerank: bool = True) -> list[dict]:
    """
    Hybrid search with RRF fusion and optional reranking.

    Args:
        query: User's natural language question.
        top_k: Number of final results to return.
        rerank: Whether to apply cross-encoder reranking.

    Returns:
        List of dicts with 'text', 'score', and 'metadata'.
    """

    # Step 1: Embed the query with BOTH models
    dense_vector = dense_model.encode(query).tolist()
    sparse_embedding = list(sparse_model.embed([query]))[0]
    sparse_vector = models.SparseVector(
        indices=sparse_embedding.indices.tolist(),
        values=sparse_embedding.values.tolist(),
    )

    # Step 2: Hybrid search - Qdrant handles EVERYTHING
    #   - Runs dense search (top 20 by semantic similarity)
    #   - Runs sparse search (top 20 by keyword match)
    #   - Fuses both lists using Reciprocal Rank Fusion
    #   - Returns top results from the fused list
    prefetch_limit = top_k * 4  # cast a wide net before reranking

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=sparse_vector,
                using="sparse",
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=prefetch_limit if rerank else top_k,
    )

    # Step 3: Format results
    candidates = [
        {
            "text": hit.payload["text"],
            "score": hit.score,
            "metadata": {
                "source": hit.payload["source"],
                "page": hit.payload["page"],
                "chunk_index": hit.payload["chunk_index"],
            },
        }
        for hit in results.points
    ]

    # Step 4: Rerank with cross-encoder (optional but recommended)
    if rerank and candidates:
        pairs = [(query, c["text"]) for c in candidates]
        rerank_scores = reranker.predict(pairs)

        for candidate, score in zip(candidates, rerank_scores):
            candidate["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    return candidates[:top_k]