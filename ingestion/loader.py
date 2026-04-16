"""
Document loader for the RAG pipeline.
Reads PDF files and extracts text with page-level metadata.
"""

from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path: str) -> list[dict]:
    """
    Load a PDF and return a list of documents, one per page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of dicts, each with 'text' and 'metadata' keys.
    """

    # Step 1: Validate the file exists before doing anything
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    # Step 2: Open and parse the PDF
    reader = PdfReader(file_path)
    total_pages = len(reader.pages)

    # Step 3: Extract text from each page
    documents = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        # Skip pages that yielded no text (likely scanned images)
        if not text.strip():
            continue

        documents.append({
            "text": text.strip(),
            "metadata": {
                "source": path.name,
                "page": page_num,
                "total_pages": total_pages,
            },
        })

    if not documents:
        raise ValueError(
            f"No text extracted from {path.name}. "
            "The PDF may be scanned (image-based). OCR is needed."
        )

    return documents

