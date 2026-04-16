"""
Text chunker for the RAG pipeline.
Splits page-level documents into smaller chunks for embedding.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(documents: list[dict], chunk_size: int = 512, chunk_overlap: int = 50) -> list[dict]:
    """
    Split page-level documents into smaller chunks, preserving metadata.

    Args:
        documents: Output from loader.py - list of {"text", "metadata"} dicts.
        chunk_size: Max tokens per chunk.
        chunk_overlap: Overlapping tokens between consecutive chunks.

    Returns:
        List of dicts with 'text' and 'metadata' (now includes chunk_index).
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for doc in documents:
        splits = splitter.split_text(doc["text"])

        for i, split_text in enumerate(splits):
            chunks.append({
                "text": split_text,
                "metadata": {
                    **doc["metadata"],         # carry forward source, page, total_pages
                    "chunk_index": i,          # which chunk within this page
                    "chunk_total": len(splits), # how many chunks this page produced
                },
            })

    return chunks
