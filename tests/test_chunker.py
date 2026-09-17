"""Tests for recursive character chunking."""

from app.chunker import chunk_text
from app.config import CHUNK_OVERLAP, CHUNK_SIZE


def test_short_text_is_single_chunk() -> None:
    chunks = chunk_text("Employees receive 20 days of annual leave.", "policy.txt")
    assert len(chunks) == 1
    assert chunks[0].position == 0
    assert chunks[0].document_name == "policy.txt"
    assert chunks[0].chunk_id == "policy.txt::chunk-0"


def test_long_text_is_split_to_configured_size() -> None:
    text = "Paragraph one. " * 200
    chunks = chunk_text(text, "long.txt")
    assert len(chunks) > 1
    assert all(len(chunk.text) <= CHUNK_SIZE + 20 for chunk in chunks)


def test_chunk_overlap_is_present() -> None:
    words = [f"word{i:03d}" for i in range(300)]
    text = " ".join(words)
    chunks = chunk_text(text, "overlap.txt", chunk_size=80, chunk_overlap=20)
    assert len(chunks) >= 2

    first_tail = chunks[0].text[-20:]
    assert any(token in chunks[1].text for token in first_tail.split() if token)


def test_chunk_size_and_overlap_constants() -> None:
    assert CHUNK_SIZE == 800
    assert CHUNK_OVERLAP == 120


def test_empty_text_returns_no_chunks() -> None:
    assert chunk_text("   ", "empty.txt") == []
