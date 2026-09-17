"""Latency tracking and remaining RAG validation tests."""

from pathlib import Path

from app.config import INSUFFICIENT_EVIDENCE_MESSAGE
from app.embeddings import embedding_dimension
from app.rag import answer_question, ingest_document, retrieve_relevant_chunks
from app.vector_store import FaissVectorStore

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "documents" / "sample_leave_policy.txt"


def _store(tmp_path: Path) -> FaissVectorStore:
    store = FaissVectorStore(
        dimension=embedding_dimension(),
        index_path=tmp_path / "index.faiss",
        metadata_path=tmp_path / "metadata.json",
    )
    ingest_document("sample_leave_policy.txt", SAMPLE_PATH.read_bytes(), store=store)
    return store


def test_retrieval_logs_and_records_component_latencies(tmp_path: Path, caplog) -> None:
    store = _store(tmp_path)
    with caplog.at_level("INFO", logger="app.rag"):
        retrieved = retrieve_relevant_chunks(
            "How many annual leave days do employees receive?",
            store=store,
        )
    assert retrieved.embedding_latency_ms > 0
    assert retrieved.retrieval_latency_ms >= 0
    assert any("embedding latency" in message for message in caplog.messages)


def test_known_and_unknown_query_latencies(tmp_path: Path, caplog) -> None:
    store = _store(tmp_path)

    def fake_llm(question, hits):
        return "Employees receive 20 days of annual leave per calendar year.", 7.5

    with caplog.at_level("INFO", logger="app.rag"):
        known = answer_question(
            "How many annual leave days do employees receive?",
            store=store,
            generate_fn=fake_llm,
        )
        unknown = answer_question(
            "What is the company's maternity leave policy?",
            store=store,
            generate_fn=fake_llm,
        )

    assert "20 days" in known.answer
    assert known.llm_latency_ms == 7.5
    assert known.total_latency_ms >= known.embedding_latency_ms
    assert unknown.answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert unknown.llm_latency_ms == 0.0
    assert unknown.sources == []
    assert any("llm=7.5" in message for message in caplog.messages)
    assert any("no LLM call" in message for message in caplog.messages)


def test_pydantic_constraints_match_api_contract() -> None:
    from app.models import QueryRequest
    from pydantic import ValidationError
    import pytest

    QueryRequest(question="How many leave days?", top_k=1)
    QueryRequest(question="How many leave days?", top_k=10)
    with pytest.raises(ValidationError):
        QueryRequest(question="no", top_k=5)
    with pytest.raises(ValidationError):
        QueryRequest(question="x" * 1001, top_k=5)
