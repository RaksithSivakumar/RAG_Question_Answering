"""SentenceTransformer embedding wrapper."""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be generated."""


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    logger.info("Loading embedding model %s", model_name)
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model: SentenceTransformer | None = None) -> np.ndarray:
    if not texts:
        raise EmbeddingError("No texts provided for embedding.")
    encoder = model or load_embedding_model()
    vectors = encoder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    array = np.asarray(vectors, dtype="float32")
    if array.ndim == 1:
        array = np.expand_dims(array, axis=0)
    return array


def embed_query(text: str, model: SentenceTransformer | None = None) -> np.ndarray:
    if not text or not text.strip():
        raise EmbeddingError("Query text is empty.")
    return embed_texts([text], model=model)


def embedding_dimension(model: SentenceTransformer | None = None) -> int:
    encoder = model or load_embedding_model()
    # Newer sentence-transformers renamed this method; keep a fallback.
    if hasattr(encoder, "get_embedding_dimension"):
        return int(encoder.get_embedding_dimension())
    return int(encoder.get_sentence_embedding_dimension())
