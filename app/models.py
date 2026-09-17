"""Pydantic request and response models."""

from pydantic import BaseModel, Field

from app.config import DEFAULT_TOP_K, MAX_TOP_K


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class SourceChunk(BaseModel):
    chunk_id: str
    document: str
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    latency_ms: float


class UploadResponse(BaseModel):
    filename: str
    chunks_created: int
    status: str
