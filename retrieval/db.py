"""
Shared resources for the RAG pipeline.
Single source of truth for client, models, and collection config.
"""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding

client = QdrantClient(path="./qdrant_data")

# Embedding models 
dense_model = SentenceTransformer("all-MiniLM-L6-v2")          # 384-dim
sparse_model = SparseTextEmbedding("Qdrant/bm25")              # BM25 sparse vectors
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") # cross-encoder reranker

#Collection 
COLLECTION_NAME = "pharma_docs"
