"""FastAPI application entrypoint (endpoints added in later milestones)."""

from fastapi import FastAPI

app = FastAPI(
    title="RAG Question Answering",
    description="Minimal explainable RAG system for grounded document Q&A.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
