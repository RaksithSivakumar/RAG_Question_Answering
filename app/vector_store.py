"""Local FAISS vector store with JSON metadata persistence."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from app.config import FAISS_INDEX_PATH, FAISS_METADATA_PATH, INDEX_DIR

logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """Raised when the FAISS index or metadata cannot be used."""


@dataclass
class ChunkMetadata:
    vector_id: int
    chunk_id: str
    document_name: str
    chunk_text: str
    chunk_position: int


@dataclass
class SearchHit:
    chunk_id: str
    document_name: str
    chunk_text: str
    chunk_position: int
    score: float


class FaissVectorStore:
    def __init__(
        self,
        dimension: int,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = FAISS_METADATA_PATH,
    ) -> None:
        if dimension <= 0:
            raise VectorStoreError("Embedding dimension must be positive.")
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: list[ChunkMetadata] = []
        self._load_if_available()

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    def add(self, embeddings: np.ndarray, records: list[ChunkMetadata]) -> int:
        vectors = _as_float32_matrix(embeddings, self.dimension)
        if vectors.shape[0] != len(records):
            raise VectorStoreError("Number of embeddings does not match metadata records.")

        start_id = self.size
        for offset, record in enumerate(records):
            record.vector_id = start_id + offset

        self.index.add(vectors)
        self.metadata.extend(records)
        self.save()
        logger.info("Added %s vectors. Index size is now %s.", vectors.shape[0], self.size)
        return vectors.shape[0]

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchHit]:
        if self.size == 0:
            return []
        if top_k <= 0:
            raise VectorStoreError("top_k must be positive.")

        query = _as_float32_matrix(query_embedding, self.dimension)
        k = min(top_k, self.size)
        scores, indices = self.index.search(query, k)

        hits: list[SearchHit] = []
        for score, vector_id in zip(scores[0], indices[0], strict=True):
            if vector_id < 0:
                continue
            record = self._metadata_by_id(int(vector_id))
            hits.append(
                SearchHit(
                    chunk_id=record.chunk_id,
                    document_name=record.document_name,
                    chunk_text=record.chunk_text,
                    chunk_position=record.chunk_position,
                    score=float(score),
                )
            )
        return hits

    def save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        payload = {
            "dimension": self.dimension,
            "metadata": [asdict(item) for item in self.metadata],
        }
        self.metadata_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def _load_if_available(self) -> None:
        if not self.index_path.exists() or not self.metadata_path.exists():
            return

        loaded = faiss.read_index(str(self.index_path))
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        stored_dimension = int(payload.get("dimension", loaded.d))

        if loaded.d != self.dimension or stored_dimension != self.dimension:
            raise VectorStoreError(
                f"Index dimension {loaded.d} does not match embedding dimension {self.dimension}."
            )
        if loaded.ntotal != len(payload.get("metadata", [])):
            raise VectorStoreError("FAISS index size does not match metadata records.")

        self.index = loaded
        self.metadata = [ChunkMetadata(**item) for item in payload["metadata"]]
        logger.info("Loaded FAISS index with %s vectors.", self.size)

    def _metadata_by_id(self, vector_id: int) -> ChunkMetadata:
        if vector_id < 0 or vector_id >= len(self.metadata):
            raise VectorStoreError(f"No metadata for vector id {vector_id}.")
        record = self.metadata[vector_id]
        if record.vector_id != vector_id:
            match = next((item for item in self.metadata if item.vector_id == vector_id), None)
            if match is None:
                raise VectorStoreError(f"No metadata for vector id {vector_id}.")
            return match
        return record


def _as_float32_matrix(embeddings: np.ndarray, expected_dimension: int) -> np.ndarray:
    array = np.asarray(embeddings, dtype="float32")
    if array.ndim == 1:
        array = np.expand_dims(array, axis=0)
    if array.ndim != 2:
        raise VectorStoreError("Embeddings must be a 1-D or 2-D array.")
    if array.shape[1] != expected_dimension:
        raise VectorStoreError(
            f"Embedding dimension {array.shape[1]} does not match index dimension {expected_dimension}."
        )
    faiss.normalize_L2(array)
    return array
