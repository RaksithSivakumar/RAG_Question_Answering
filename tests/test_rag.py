"""Retrieval tests against the sample leave-policy document."""

from pathlib import Path

from app.config import INSUFFICIENT_EVIDENCE_MESSAGE, RELEVANCE_THRESHOLD
from app.embeddings import embedding_dimension
from app.rag import ingest_document, retrieve_relevant_chunks
from app.vector_store import FaissVectorStore

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "documents" / "sample_leave_policy.txt"


def _indexed_store(tmp_path: Path) -> FaissVectorStore:
    store = FaissVectorStore(
        dimension=embedding_dimension(),
        index_path=tmp_path / "index.faiss",
        metadata_path=tmp_path / "metadata.json",
    )
    ingest_document("sample_leave_policy.txt", SAMPLE_PATH.read_bytes(), store=store)
    return store


def test_ingest_creates_chunks(tmp_path: Path) -> None:
    store = _indexed_store(tmp_path)
    assert store.size >= 1


def test_known_question_retrieves_annual_leave(tmp_path: Path) -> None:
    store = _indexed_store(tmp_path)
    result = retrieve_relevant_chunks(
        "How many annual leave days do employees receive?",
        top_k=5,
        store=store,
    )
    assert result.insufficient_evidence is False
    assert result.hits
    combined = " ".join(hit.chunk_text for hit in result.hits)
    assert "20 days" in combined


def test_unknown_question_is_insufficient(tmp_path: Path) -> None:
    store = _indexed_store(tmp_path)
    result = retrieve_relevant_chunks(
        "What is the company's maternity leave policy?",
        top_k=5,
        store=store,
    )
    # Either filtered out by the relevance threshold, or no supporting text.
    combined = " ".join(hit.chunk_text.lower() for hit in result.hits)
    assert "maternity" not in combined
    if result.insufficient_evidence:
        assert result.message == INSUFFICIENT_EVIDENCE_MESSAGE
        assert result.hits == []


def test_relevance_threshold_is_configurable() -> None:
    assert 0.0 < RELEVANCE_THRESHOLD < 1.0
