# Pharma RAG Q&A

A domain-specific question answering system built on top of Bayer crop science product labels. Uses hybrid retrieval (dense + sparse vectors), reciprocal rank fusion, cross-encoder reranking, and grounded generation with source citations.

Live demo: `[your-app-runner-url]`

## What it does

Upload a pharma or agriculture PDF. The system chunks it, embeds it using both semantic and keyword representations, stores it in Qdrant, and then answers natural language questions with cited sources. Answers are grounded strictly in the retrieved context to minimize hallucination.

## Architecture

```
PDF --> Loader --> Chunker --> Embedder --> Qdrant (dense + sparse vectors)
                                              |
User query --> Hybrid Search (RRF fusion) --> Reranker --> Prompt Builder --> LLM --> Cited Answer
```

**Ingestion**
- PDF text extraction with page-level metadata tracking
- Recursive character splitting (512 chars, 50 overlap) to preserve semantic boundaries
- Dual embedding: `all-MiniLM-L6-v2` for dense vectors (384-dim), `Qdrant/BM25` for sparse term vectors

**Retrieval**
- Qdrant's native prefetch runs both dense and sparse searches in a single call
- Reciprocal Rank Fusion merges the two ranked lists without manual score normalization
- Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) reorders the top-20 candidates for precision

**Generation**
- GPT-4o-mini at temperature 0 for deterministic, reproducible answers
- System prompt enforces context-only grounding with `[Page X]` citation format
- Response includes answer, source list, and latency in milliseconds

**Evaluation**
- 10 golden questions with expected answers covering factual recall, comparison, and cross-document reasoning
- 4 metrics via DeepEval: Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall
- Judge model: Gemini 2.5 Flash Lite
- All metrics scored against a 0.7 threshold

## Project structure

```
pharma-rag-qa/
├── api.py                  # FastAPI server with /query and /eval endpoints
├── app.py                  # Gradio UI with file upload (local dev)
├── app_hosted.py           # Gradio UI without upload (Docker/hosted)
├── generation/
│   └── chain.py            # RAG orchestrator: retrieve -> prompt -> generate
├── ingestion/
│   ├── loader.py           # PDF to page-level text with metadata
│   ├── chunker.py          # Recursive text splitting via LangChain
│   └── embedder.py         # Dense + sparse embedding, Qdrant storage
├── retrieval/
│   ├── db.py               # Shared Qdrant client, models, collection config
│   └── search.py           # Hybrid search + RRF + cross-encoder reranking
├── eval/
│   ├── golden_dataset.py   # Test questions and expected answers
│   ├── test_rag.py         # Evaluation runner with per-metric error handling
│   └── results.json        # Pre-computed scores
├── DTO/
│   └── request_response_models.py
├── scripts/
│   ├── pre_compute.py      # One-time ingestion into persistent Qdrant
│   └── download_models.py  # Pre-download HF models for Docker build
├── qdrant_data/            # Pre-computed vectors (persistent Qdrant storage)
├── Dockerfile              # Multi-stage build: builder + runtime
└── terraform/
    └── main.tf             # Azure Container Instance IaC
```

## Design decisions

**RAG over fine-tuning**: Product label data changes with regulatory updates. RAG gives source attribution out of the box and avoids the 6-8 week fine-tuning cycle. For stable, task-specific behavior (like Bayer's E.L.Y. crop advisor), fine-tuning makes more sense.

**Hybrid search over dense-only**: Pharma documents have domain-specific terms (IRAC groups, active ingredient concentrations, PHI values) that need exact keyword matching. Dense search alone misses these. BM25 catches them. RRF combines both without needing score normalization.

**Reranking on top of fusion**: Bi-encoders embed query and document separately, which is fast but approximate. The cross-encoder scores query-document pairs jointly, catching relevance signals that bi-encoders miss. Running it on 20 candidates instead of the full corpus keeps latency under 300ms.

**Pre-computed embeddings in Docker**: The container starts in seconds instead of re-ingesting on every deploy. Models are baked into the image during build to avoid cold-start downloads. Qdrant runs in persistent local mode with no external database dependency.

**Evaluation as a first-class feature**: Most RAG systems ship without measurable quality metrics. The eval suite runs 10 golden questions through the live pipeline and scores faithfulness, relevancy, precision, and recall. Results are served in the UI and via API so quality is always visible.

## Running locally

```bash
# Install deps
pip install -r requirements.txt

# Set your OpenAI key
export OPENAI_API_KEY=sk-your-key

# Option 1: Ingest a new document and run
python scripts/pre_compute.py "path/to/your.pdf"
python app.py
# Open http://localhost:7860

# Option 2: API server
python api.py
# Open http://localhost:8000/docs for Swagger UI
```

## Running with Docker

```bash
docker build -t pharma-rag-qa .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-your-key pharma-rag-qa
```

## Running evaluation

```bash
# Needs OPENAI_API_KEY and GEMINI_API_KEY set
python eval/test_rag.py
# Results saved to eval/results.json
```

## Tech stack

| Layer | Tool | Why |
|-------|------|-----|
| Embedding (dense) | all-MiniLM-L6-v2 | Runs on CPU, no API dependency, 384-dim |
| Embedding (sparse) | Qdrant/BM25 via fastembed | Exact term matching for domain vocabulary |
| Vector store | Qdrant (local persistent) | Native hybrid search + RRF, no infra overhead |
| Fusion | Reciprocal Rank Fusion | Merges ranked lists without score normalization |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Joint query-doc scoring on top-k candidates |
| LLM | GPT-4o-mini | Fast, cheap, deterministic at temp 0 |
| Orchestration | LangChain (LCEL) | Composable chains, swappable components |
| API | FastAPI | Auto-generated docs, Pydantic validation |
| UI | Gradio | Native chat components, quick to build |
| Evaluation | DeepEval + RAGAS | Standard RAG metrics with LLM-as-judge |
| Containerization | Docker (multi-stage) | Builder stage compiles deps, runtime stage is lean |
| IaC | Terraform | Azure Container Instance deployment |
