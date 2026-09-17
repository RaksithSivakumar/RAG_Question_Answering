"""Grounded retrieval pipeline: ingest documents and retrieve relevant chunks."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.chunker import chunk_text
from app.config import (
    DEFAULT_TOP_K,
    INSUFFICIENT_EVIDENCE_MESSAGE,
    MAX_TOP_K,
    RELEVANCE_THRESHOLD,
)
from app.embeddings import embed_query, embed_texts, embedding_dimension
from app.llm import generate_grounded_answer
from app.loaders import extract_text_from_bytes
from app.vector_store import ChunkMetadata, FaissVectorStore, SearchHit

logger = logging.getLogger(__name__)

_store: FaissVectorStore | None = None


@dataclass
class IngestResult:
    filename: str
    chunks_created: int
    status: str = "indexed"


@dataclass
class RetrievalResult:
    hits: list[SearchHit] = field(default_factory=list)
    insufficient_evidence: bool = False
    embedding_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0

    @property
    def message(self) -> str | None:
        if self.insufficient_evidence:
            return INSUFFICIENT_EVIDENCE_MESSAGE
        return None


@dataclass
class QueryResult:
    answer: str
    sources: list[SearchHit]
    embedding_latency_ms: float
    retrieval_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float


def get_vector_store() -> FaissVectorStore:
    global _store
    if _store is None:
        _store = FaissVectorStore(dimension=embedding_dimension())
    return _store


def set_vector_store(store: FaissVectorStore | None) -> None:
    """Replace the process-wide store (used by tests)."""
    global _store
    _store = store


def ingest_document(
    filename: str,
    content: bytes,
    store: FaissVectorStore | None = None,
) -> IngestResult:
    text = extract_text_from_bytes(filename, content)
    chunks = chunk_text(text, document_name=filename)
    if not chunks:
        raise ValueError("No chunks were produced from the document.")

    vectors = embed_texts([chunk.text for chunk in chunks])
    records = [
        ChunkMetadata(
            vector_id=-1,
            chunk_id=chunk.chunk_id,
            document_name=chunk.document_name,
            chunk_text=chunk.text,
            chunk_position=chunk.position,
        )
        for chunk in chunks
    ]
    target = store if store is not None else get_vector_store()
    target.add(vectors, records)
    logger.info("Indexed %s chunks from %s", len(chunks), filename)
    return IngestResult(filename=filename, chunks_created=len(chunks))


def retrieve_relevant_chunks(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    store: FaissVectorStore | None = None,
    threshold: float = RELEVANCE_THRESHOLD,
) -> RetrievalResult:
    if top_k < 1 or top_k > MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")

    target = store if store is not None else get_vector_store()
    embed_started = time.perf_counter()
    query_vector = embed_query(question)
    embedding_latency_ms = (time.perf_counter() - embed_started) * 1000

    search_started = time.perf_counter()
    raw_hits = target.search(query_vector, top_k=top_k)
    retrieval_latency_ms = (time.perf_counter() - search_started) * 1000

    filtered = [hit for hit in raw_hits if hit.score >= threshold]
    insufficient = len(filtered) == 0
    if insufficient:
        logger.info(
            "Insufficient evidence for question (raw_hits=%s, threshold=%.2f)",
            len(raw_hits),
            threshold,
        )
    return RetrievalResult(
        hits=filtered,
        insufficient_evidence=insufficient,
        embedding_latency_ms=embedding_latency_ms,
        retrieval_latency_ms=retrieval_latency_ms,
    )


def answer_question(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    store: FaissVectorStore | None = None,
    threshold: float = RELEVANCE_THRESHOLD,
    generate_fn=generate_grounded_answer,
) -> QueryResult:
    started = time.perf_counter()
    retrieved = retrieve_relevant_chunks(question, top_k=top_k, store=store, threshold=threshold)

    if retrieved.insufficient_evidence:
        total_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "query latencies ms: embedding=%.1f retrieval=%.1f llm=0.0 total=%.1f (no LLM call)",
            retrieved.embedding_latency_ms,
            retrieved.retrieval_latency_ms,
            total_ms,
        )
        return QueryResult(
            answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            sources=[],
            embedding_latency_ms=retrieved.embedding_latency_ms,
            retrieval_latency_ms=retrieved.retrieval_latency_ms,
            llm_latency_ms=0.0,
            total_latency_ms=total_ms,
        )

    answer, llm_latency_ms = generate_fn(question, retrieved.hits)
    total_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "query latencies ms: embedding=%.1f retrieval=%.1f llm=%.1f total=%.1f",
        retrieved.embedding_latency_ms,
        retrieved.retrieval_latency_ms,
        llm_latency_ms,
        total_ms,
    )
    return QueryResult(
        answer=answer,
        sources=retrieved.hits,
        embedding_latency_ms=retrieved.embedding_latency_ms,
        retrieval_latency_ms=retrieved.retrieval_latency_ms,
        llm_latency_ms=llm_latency_ms,
        total_latency_ms=total_ms,
    )
