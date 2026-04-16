from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, examples=["What is Bayer's AI platform?"])
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = Field(default=True)


class Source(BaseModel):
    source: str
    page: int
    score: float
    preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: float