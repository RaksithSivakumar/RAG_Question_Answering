"""Shared configuration constants for the RAG pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
INDEX_DIR = DATA_DIR / "index"

# Recursive character chunking. These are starting values, not universally optimal.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# SentenceTransformer embedding model (384-dimensional vectors).
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# FAISS persistence paths
FAISS_INDEX_PATH = INDEX_DIR / "index.faiss"
FAISS_METADATA_PATH = INDEX_DIR / "metadata.json"

# Retrieval: empirical starting threshold on cosine similarity in [0, 1].
# Tune later with a labelled evaluation set.
RELEVANCE_THRESHOLD = 0.45
DEFAULT_TOP_K = 5
MAX_TOP_K = 10

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I don't have enough information in the provided documents to answer this question."
)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}

# Gemini (never log the API key)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "30"))
