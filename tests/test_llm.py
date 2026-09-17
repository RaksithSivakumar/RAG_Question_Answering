"""Tests for grounded Gemini generation and error handling."""

from types import SimpleNamespace

import pytest

from app.config import INSUFFICIENT_EVIDENCE_MESSAGE
from app.llm import GROUNDED_SYSTEM_INSTRUCTION, LLMConfigError, LLMError, generate_grounded_answer
from app.rag import answer_question
from app.vector_store import SearchHit


def _hit() -> SearchHit:
    return SearchHit(
        chunk_id="policy.txt::chunk-0",
        document_name="policy.txt",
        chunk_text="Employees receive 20 days of annual leave per calendar year.",
        chunk_position=0,
        score=0.81,
    )


def test_missing_api_key_raises_config_error() -> None:
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        generate_grounded_answer("How many leave days?", [_hit()], api_key="")


def test_empty_hits_skip_llm() -> None:
    answer, latency = generate_grounded_answer("Anything?", [])
    assert answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert latency == 0.0


def test_empty_model_response_is_rejected() -> None:
    class EmptyClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                return SimpleNamespace(text="   ")

    with pytest.raises(LLMError, match="empty response"):
        generate_grounded_answer("How many leave days?", [_hit()], api_key="test-key", client=EmptyClient())


def test_api_failure_does_not_expose_secret() -> None:
    class FailingClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                raise RuntimeError("secret-api-key-should-not-leak")

    with pytest.raises(LLMError, match="language model request failed"):
        generate_grounded_answer("How many leave days?", [_hit()], api_key="secret-api-key-should-not-leak", client=FailingClient())


def test_prompt_requires_grounding() -> None:
    assert "Answer ONLY using the supplied context" in GROUNDED_SYSTEM_INSTRUCTION
    assert "Do not use outside knowledge" in GROUNDED_SYSTEM_INSTRUCTION


def test_answer_question_skips_llm_when_index_empty(tmp_path, monkeypatch) -> None:
    from app.embeddings import embedding_dimension
    from app.vector_store import FaissVectorStore

    store = FaissVectorStore(
        dimension=embedding_dimension(),
        index_path=tmp_path / "index.faiss",
        metadata_path=tmp_path / "metadata.json",
    )

    def boom(*args, **kwargs):
        raise AssertionError("LLM should not be called without evidence")

    result = answer_question(
        "What is the company's maternity leave policy?",
        store=store,
        generate_fn=boom,
    )
    assert result.answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert result.sources == []
    assert result.llm_latency_ms == 0.0
