"""
Pre-compute: load, chunk, embed, store in persistent Qdrant.
Run ONCE locally before building Docker image.

Usage: python scripts/precompute.py "path/to/your.pdf"
"""


import argparse

from ingestion.loader import load_pdf
from ingestion.chunker import create_chunks
from ingestion.embedder import create_collection, embed_and_store
from retrieval.db import client, COLLECTION_NAME

def precompute(file_path: str):
    print(f"Loading {file_path}...")
    docs = load_pdf(file_path)
    print(f"  {len(docs)} pages extracted")

    print("Chunking...")
    chunks = create_chunks(docs)
    print(f"  {len(chunks)} chunks created")

    print("Creating collection...")
    create_collection()

    print("Embedding and storing...")
    count = embed_and_store(chunks)
    print(f"  {count} points stored in Qdrant")

    # Verify it worked
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"\nVerification: {collection_info.points_count} points in collection")

    client.close()
    print("Qdrant data flushed to disk. Done.")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute RAG pipeline")
    parser.add_argument("pdf_path", help="Path to PDF file to ingest")
    args = parser.parse_args()
    precompute(args.pdf_path)