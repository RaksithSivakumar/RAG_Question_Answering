# RAG Question Answering

Minimal, explainable Retrieval-Augmented Generation (RAG) system. A user uploads PDF or TXT documents, asks a question, and receives an answer grounded in retrieved chunks rather than the model's general knowledge.

This project does **not** use LangChain, LlamaIndex, CrewAI, Pinecone, or Chroma. Retrieval and generation are implemented directly with FastAPI, SentenceTransformers, FAISS, and the Google Gemini API.

## Architecture

1. **Load** — PDF (PyMuPDF) or TXT bytes are validated and converted to text.
2. **Chunk** — Recursive character splitting with configurable size and overlap.
3. **Embed** — `all-MiniLM-L6-v2` produces 384-dimensional vectors.
4. **Index** — FAISS `IndexFlatIP` stores L2-normalized vectors (cosine similarity) plus JSON metadata.
5. **Retrieve** — The question is embedded and the top-k neighbours are filtered by a similarity threshold.
6. **Generate** — If at least one chunk passes the threshold, Gemini answers using only that context. Otherwise the API returns a fixed insufficient-evidence message and **does not call** the LLM.

```
Upload / Query  →  FastAPI  →  loaders / chunker / embeddings / FAISS  →  Gemini (optional)
```

## Features

- `POST /documents/upload` for `.pdf` and `.txt`
- `POST /query` with Pydantic validation
- Persistent local FAISS index and chunk metadata
- Source chunks returned with similarity scores
- Unknown-question handling without using outside knowledge
- Component latency logging (embedding, retrieval, LLM, total)

Not implemented: frontend, authentication, database, multi-user isolation, labelled RAG evaluation suite.

## Tech stack

| Layer | Library |
| --- | --- |
| API | FastAPI, Pydantic |
| PDF | PyMuPDF |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS (`faiss-cpu`) |
| LLM | google-genai (Gemini) |
| Tests | pytest, httpx |

## Project structure

```
rag-question-answering/
├── app/
│   ├── main.py              # FastAPI routes
│   ├── models.py            # Request/response schemas
│   ├── loaders.py           # PDF/TXT extraction
│   ├── chunker.py           # Recursive character chunking
│   ├── embeddings.py        # SentenceTransformer wrapper
│   ├── vector_store.py      # FAISS + metadata persistence
│   ├── llm.py               # Grounded Gemini generation
│   ├── rag.py               # Ingest / retrieve / answer
│   └── config.py            # Tunable constants
├── data/documents/          # Sample policy TXT
├── data/index/              # Generated FAISS files (gitignored)
├── tests/
├── explanation/explanation.md
├── .env.example
├── requirements.txt
└── requirements-dev.txt
```

## Setup

Python 3.11+ recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
copy .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY`. Do not commit `.env`.

The first embedding call downloads `all-MiniLM-L6-v2` from Hugging Face (network required once).

## Environment variables

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Required for answer generation |
| `GEMINI_MODEL` | Defaults to `gemini-3.6-flash` |
| `GEMINI_TIMEOUT_SECONDS` | HTTP timeout (default 30) |

The API key is never included in HTTP error bodies.

## Running

From the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs for the interactive schema.

## API endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Process liveness |
| `POST` | `/documents/upload` | Multipart file upload (`.pdf` / `.txt`) |
| `POST` | `/query` | Grounded question answering |

### Example upload

```powershell
curl.exe -X POST http://127.0.0.1:8000/documents/upload -F "file=@data/documents/sample_leave_policy.txt"
```

Example response (verified locally):

```json
{
  "filename": "sample_leave_policy.txt",
  "chunks_created": 1,
  "status": "indexed"
}
```

### Example query

Request constraints: `question` length 3–1000; `top_k` 1–10.

```json
{
  "question": "How many annual leave days do employees receive?",
  "top_k": 5
}
```

Verified response shape (17 Sep 2026, local Windows machine, `gemini-3.6-flash`):

```json
{
  "answer": "Employees receive 20 days of annual leave per calendar year.",
  "sources": [
    {
      "chunk_id": "sample_leave_policy.txt::chunk-0",
      "document": "sample_leave_policy.txt",
      "text": "Employees receive 20 days of annual leave per calendar year. ...",
      "score": 0.768
    }
  ],
  "latency_ms": 3423.8
}
```

Unsupported extensions and empty extracted text return HTTP 400. Invalid query payloads return HTTP 422. Missing Gemini configuration returns HTTP 503. Gemini call failures return HTTP 502 without leaking secrets.

## Chunking strategy

Recursive character splitting uses separators `\n\n`, `\n`, `. `, ` `, then characters.

- `CHUNK_SIZE = 800`
- `CHUNK_OVERLAP = 120`

These live in `app/config.py`. They were chosen as a starting point: 800 characters is small enough for MiniLM's short-passage sweet spot and large enough to keep a policy paragraph intact; 120 characters (~15%) overlap reduces boundary splits. **They are not claimed to be universally optimal.** A labelled chunking study would be required to justify other values.

## Retrieval strategy

- Query embedding with the same MiniLM model (normalized).
- FAISS inner-product search ≈ cosine similarity in `[−1, 1]` (typically `[0, 1]` for related text).
- `RELEVANCE_THRESHOLD = 0.45` (config constant). This is an **empirical starting point**, not a tuned production threshold. It should be calibrated on a labelled evaluation set.

Measured on the sample leave policy (same session as above):

| Question | Top-1 cosine | Action |
| --- | --- | --- |
| Annual leave days | 0.7680 | Kept; LLM called |
| Maternity leave policy | 0.3637 | Dropped; no LLM |
| Capital of France | −0.0130 | Dropped; no LLM |

## Unknown-question handling

If no neighbour is ≥ `RELEVANCE_THRESHOLD`, the system returns:

> I don't have enough information in the provided documents to answer this question.

and skips Gemini. Verified for *“What is the company's maternity leave policy?”* against the sample document (no maternity facts). The grounded prompt is a second line of defence if a weakly related chunk were to pass the threshold.

## Error handling

| Situation | Behaviour |
| --- | --- |
| Wrong file extension | 400 validation error |
| Empty file / empty extracted text | 400 |
| Query too short/long or bad `top_k` | 422 |
| FAISS dimension mismatch | `VectorStoreError` → 500 |
| Missing `GEMINI_API_KEY` | 503 |
| Gemini auth / 404 model / timeout / empty output | 502, generic message |

## Metrics

Python logging records:

- embedding latency
- retrieval latency
- LLM latency (0 when skipped)
- end-to-end total

### Measured results (fill-in after testing)

Recorded 17 Sep 2026 on this development machine. Not a published benchmark. First request also loads the embedding model.

| Scenario | `latency_ms` (API) | Notes |
| --- | --- | --- |
| Known question after upload | 3423.8 | Includes Gemini (~3.1 s in a prior script) |
| Unknown / out-of-document question | 18.8 | LLM skipped; sources `[]` |

Add further runs below after your own hardware tests.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Last full run during development: **37 passed**.

Coverage includes TXT/PDF loading, empty documents, unsupported types, chunk size/overlap, FAISS persistence and dimension mismatch, Pydantic constraints, unknown questions, and latency logging.

## Known limitations

- Single shared FAISS index per process; uploads accumulate and are not isolated per user.
- No deletion or re-index API.
- Threshold 0.45 is not validated on a labelled set.
- IndexFlatIP is exact but not scaled for large corpora.
- Scanned PDFs without a text layer extract as empty and are rejected.
- Gemini model IDs change; `gemini-2.0-flash` returned 404 during development (see explanation).
- No citation highlighting inside the generated answer beyond returned source chunks.

## Future improvements

- Labelled retrieval evaluation (precision@k, recall, hallucination rate).
- Threshold and chunk-size sweeps.
- Index rebuild / document delete.
- Hybrid lexical + dense retrieval for policy jargon.
- Persist uploads separately from the sample fixture.

## Video walkthrough placeholder

**TODO:** Record a 3–5 minute walkthrough showing:

1. Project structure and `config.py` constants.
2. Start uvicorn; upload `sample_leave_policy.txt`.
3. Ask the annual-leave question; show 20 days and source chunk.
4. Ask the maternity-leave question; show the insufficient-information reply and empty sources.
5. Point at pytest results and git log.

---

Claims in this README are limited to behaviour that was implemented and executed locally (pytest plus a live FastAPI session with Gemini).
