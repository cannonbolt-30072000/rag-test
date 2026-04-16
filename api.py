"""
FastAPI server for the RAG pipeline.
Exposes /query endpoint, health check, and serves Gradio UI.
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from dotenv import load_dotenv
load_dotenv()

from ingestion.loader import load_pdf
from ingestion.chunker import create_chunks
from ingestion.embedder import create_collection, embed_and_store
from generation.chain import query
from DTO.request_response_models import QueryRequest, QueryResponse, Source

import time
import json
from pathlib import Path
# --- Lifespan: runs once at startup ---

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Load documents into Qdrant when the server starts."""

#     print("Ingesting documents...")
#     docs = load_pdf(r"C:\Users\DELL\Desktop\Bayer_Hackathon_Practice\pdfs\Bayer_CropScience_Product_Guide.pdf")  # swap path as needed
#     chunks = create_chunks(docs)
#     create_collection()
#     count = embed_and_store(chunks)
#     print(f"Ready. {count} chunks in Qdrant.")

#     yield  # server runs

#     print("Shutting down.")

# --- App ---
app = FastAPI(
    title="Pharma RAG Q&A",
    description="Domain-specific Q&A over pharma/agriculture documents",
    version="0.1.0",
)


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    """Ask a question against the ingested documents."""

    start = time.time()

    try:
        result = query(
            question=request.question,
            top_k=request.top_k,
            rerank=request.rerank,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    latency_ms = (time.time() - start) * 1000

    return QueryResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        latency_ms=round(latency_ms, 1),
    )


@app.get("/eval")
def get_eval_results():
    """Serve pre-computed evaluation metrics."""

    eval_path = Path("eval/results.json")
    if not eval_path.exists():
        raise HTTPException(status_code=404, detail="Eval results not found. Run eval first.")

    return json.loads(eval_path.read_text())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)