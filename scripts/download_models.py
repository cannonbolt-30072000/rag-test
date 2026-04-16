"""
Pre-download all models so Docker image has them baked in.
Run during Docker build.
"""

from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding

print("Downloading dense model...")
SentenceTransformer("all-MiniLM-L6-v2")

print("Downloading sparse model...")
SparseTextEmbedding("Qdrant/bm25")

print("Downloading reranker...")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("All models cached.")