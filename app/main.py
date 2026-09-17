"""FastAPI application exposing document ingestion and grounded Q&A."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.llm import LLMConfigError, LLMError
from app.loaders import DocumentLoadError
from app.models import QueryRequest, QueryResponse, SourceChunk, UploadResponse
from app.rag import answer_question, ingest_document
from app.vector_store import VectorStoreError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("RAG Question Answering API starting")
    yield
    logger.info("RAG Question Answering API stopped")


app = FastAPI(
    title="RAG Question Answering",
    description="Minimal explainable RAG system for grounded document Q&A.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(DocumentLoadError)
async def document_load_handler(_request, exc: DocumentLoadError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(VectorStoreError)
async def vector_store_handler(_request, exc: VectorStoreError) -> JSONResponse:
    logger.error("Vector store error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "The vector index is unavailable."})


@app.exception_handler(LLMConfigError)
async def llm_config_handler(_request, exc: LLMConfigError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(LLMError)
async def llm_handler(_request, exc: LLMError) -> JSONResponse:
    logger.error("LLM error: %s", exc)
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    filename = file.filename or ""
    content = await file.read()
    result = ingest_document(filename, content)
    return UploadResponse(
        filename=result.filename,
        chunks_created=result.chunks_created,
        status=result.status,
    )


@app.post("/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest) -> QueryResponse:
    result = answer_question(payload.question, top_k=payload.top_k)
    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceChunk(
                chunk_id=hit.chunk_id,
                document=hit.document_name,
                text=hit.chunk_text,
                score=round(hit.score, 4),
            )
            for hit in result.sources
        ],
        latency_ms=round(result.total_latency_ms, 1),
    )


@app.exception_handler(ValueError)
async def value_error_handler(_request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(HTTPException)
async def http_exception_passthrough(_request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
