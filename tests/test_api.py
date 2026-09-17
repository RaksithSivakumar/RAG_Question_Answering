"""API tests for upload, query validation, and unknown-question handling."""

from pathlib import Path

import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient

from app.config import INSUFFICIENT_EVIDENCE_MESSAGE
from app.embeddings import embedding_dimension
from app.main import app
from app.rag import set_vector_store
from app.vector_store import FaissVectorStore

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "documents" / "sample_leave_policy.txt"


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    store = FaissVectorStore(
        dimension=embedding_dimension(),
        index_path=tmp_path / "index.faiss",
        metadata_path=tmp_path / "metadata.json",
    )
    set_vector_store(store)

    def fake_generate(question: str, hits):
        combined = " ".join(hit.chunk_text for hit in hits)
        if "20 days" in combined:
            return "Employees receive 20 days of annual leave per calendar year.", 5.0
        return INSUFFICIENT_EVIDENCE_MESSAGE, 5.0

    monkeypatch.setattr("app.rag.generate_grounded_answer", fake_generate)
    with TestClient(app) as test_client:
        yield test_client
    set_vector_store(None)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_validation_rejects_short_question(client: TestClient) -> None:
    response = client.post("/query", json={"question": "ab", "top_k": 5})
    assert response.status_code == 422


def test_query_validation_rejects_top_k_out_of_range(client: TestClient) -> None:
    too_low = client.post("/query", json={"question": "How many leave days?", "top_k": 0})
    too_high = client.post("/query", json={"question": "How many leave days?", "top_k": 11})
    assert too_low.status_code == 422
    assert too_high.status_code == 422


def test_unsupported_file_type(client: TestClient) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.png", b"not-a-document", "image/png")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_empty_document(client: TestClient) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("empty.txt", b"   \n", "text/plain")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_txt_upload_and_known_question(client: TestClient) -> None:
    upload = client.post(
        "/documents/upload",
        files={"file": ("sample_leave_policy.txt", SAMPLE_PATH.read_bytes(), "text/plain")},
    )
    assert upload.status_code == 200
    body = upload.json()
    assert body["filename"] == "sample_leave_policy.txt"
    assert body["chunks_created"] >= 1
    assert body["status"] == "indexed"

    query = client.post(
        "/query",
        json={"question": "How many annual leave days do employees receive?", "top_k": 5},
    )
    assert query.status_code == 200
    payload = query.json()
    assert "20 days" in payload["answer"]
    assert payload["sources"]
    assert "latency_ms" in payload
    assert payload["sources"][0]["chunk_id"]
    assert payload["sources"][0]["document"] == "sample_leave_policy.txt"


def test_unknown_question_does_not_use_outside_knowledge(client: TestClient) -> None:
    client.post(
        "/documents/upload",
        files={"file": ("sample_leave_policy.txt", SAMPLE_PATH.read_bytes(), "text/plain")},
    )
    query = client.post(
        "/query",
        json={"question": "What is the company's maternity leave policy?", "top_k": 5},
    )
    assert query.status_code == 200
    payload = query.json()
    assert payload["answer"] == INSUFFICIENT_EVIDENCE_MESSAGE
    assert payload["sources"] == []


def test_pdf_upload(client: TestClient) -> None:
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Employees receive 20 days of annual leave per calendar year.")
    content = pdf.tobytes()
    pdf.close()

    response = client.post(
        "/documents/upload",
        files={"file": ("policy.pdf", content, "application/pdf")},
    )
    assert response.status_code == 200
    assert response.json()["chunks_created"] >= 1
