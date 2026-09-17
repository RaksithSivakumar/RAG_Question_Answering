"""Tests for embeddings and FAISS persistence."""

from pathlib import Path

import numpy as np
import pytest

from app.embeddings import embed_texts, embedding_dimension
from app.vector_store import ChunkMetadata, FaissVectorStore, VectorStoreError


def test_embeddings_have_expected_dimension(tmp_path: Path) -> None:
    vectors = embed_texts(["Employees receive 20 days of annual leave."])
    assert vectors.shape[1] == embedding_dimension()
    assert vectors.dtype == np.float32


def test_faiss_add_search_and_reload(tmp_path: Path) -> None:
    dimension = 8
    store = FaissVectorStore(
        dimension=dimension,
        index_path=tmp_path / "index.faiss",
        metadata_path=tmp_path / "metadata.json",
    )
    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype="float32",
    )
    records = [
        ChunkMetadata(vector_id=-1, chunk_id="a", document_name="doc.txt", chunk_text="leave days", chunk_position=0),
        ChunkMetadata(vector_id=-1, chunk_id="b", document_name="doc.txt", chunk_text="remote work", chunk_position=1),
    ]
    store.add(vectors, records)
    hits = store.search(vectors[0], top_k=1)
    assert hits[0].chunk_id == "a"
    assert hits[0].score > 0.9

    reloaded = FaissVectorStore(
        dimension=dimension,
        index_path=tmp_path / "index.faiss",
        metadata_path=tmp_path / "metadata.json",
    )
    assert reloaded.size == 2
    assert reloaded.metadata[0].chunk_text == "leave days"


def test_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    store = FaissVectorStore(
        dimension=4,
        index_path=tmp_path / "index.faiss",
        metadata_path=tmp_path / "metadata.json",
    )
    store.add(
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"),
        [ChunkMetadata(0, "a", "doc.txt", "text", 0)],
    )
    with pytest.raises(VectorStoreError, match="does not match"):
        FaissVectorStore(
            dimension=8,
            index_path=tmp_path / "index.faiss",
            metadata_path=tmp_path / "metadata.json",
        )
