"""Grounded Gemini generation with strict context-only prompting."""

from __future__ import annotations

import logging
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TIMEOUT_SECONDS, INSUFFICIENT_EVIDENCE_MESSAGE
from app.vector_store import SearchHit

logger = logging.getLogger(__name__)

GROUNDED_SYSTEM_INSTRUCTION = """You are a document question-answering assistant.

Rules:
1. Answer ONLY using the supplied context chunks.
2. Do not use outside knowledge, training data, or general world facts.
3. Do not invent facts, numbers, policies, or names that are not in the context.
4. If the context does not contain enough information to answer the question, reply with exactly:
I don't have enough information in the provided documents to answer this question.
5. Keep the answer concise.
6. Every factual statement must be supported by the supplied context.
"""


class LLMError(RuntimeError):
    """Raised when Gemini cannot produce a grounded answer."""


class LLMConfigError(LLMError):
    """Raised when the Gemini client is not configured."""


def generate_grounded_answer(
    question: str,
    hits: list[SearchHit],
    *,
    api_key: str | None = None,
    model_name: str | None = None,
    timeout_seconds: int | None = None,
    client: object | None = None,
) -> tuple[str, float]:
    """Return (answer_text, llm_latency_ms)."""
    if not hits:
        return INSUFFICIENT_EVIDENCE_MESSAGE, 0.0

    key = GEMINI_API_KEY if api_key is None else api_key
    if not key:
        raise LLMConfigError("GEMINI_API_KEY is not configured.")

    prompt = _build_user_prompt(question, hits)
    started = time.perf_counter()
    try:
        text = _call_gemini(
            prompt,
            api_key=key,
            model_name=model_name or GEMINI_MODEL,
            timeout_seconds=timeout_seconds if timeout_seconds is not None else GEMINI_TIMEOUT_SECONDS,
            client=client,
        )
    except LLMError:
        raise
    except genai_errors.ClientError as exc:
        raise LLMError(_safe_client_message(exc)) from exc
    except genai_errors.ServerError as exc:
        raise LLMError("The language model service is currently unavailable.") from exc
    except genai_errors.APIError as exc:
        raise LLMError("The language model request failed.") from exc
    except TimeoutError as exc:
        raise LLMError("The language model request timed out.") from exc
    except Exception as exc:
        logger.exception("Unexpected Gemini failure: %s", type(exc).__name__)
        raise LLMError("The language model request failed.") from exc

    latency_ms = (time.perf_counter() - started) * 1000
    cleaned = (text or "").strip()
    if not cleaned:
        raise LLMError("The language model returned an empty response.")
    return cleaned, latency_ms


def _call_gemini(
    prompt: str,
    *,
    api_key: str,
    model_name: str,
    timeout_seconds: int,
    client: object | None,
) -> str | None:
    sdk = client or genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=timeout_seconds * 1000),
    )
    response = sdk.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=GROUNDED_SYSTEM_INSTRUCTION,
            temperature=0.0,
            http_options=types.HttpOptions(timeout=timeout_seconds * 1000),
        ),
    )
    return getattr(response, "text", None)


def _build_user_prompt(question: str, hits: list[SearchHit]) -> str:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        blocks.append(
            f"[Chunk {index} | id={hit.chunk_id} | document={hit.document_name} | score={hit.score:.3f}]\n{hit.chunk_text}"
        )
    context = "\n\n".join(blocks)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )


def _safe_client_message(exc: genai_errors.ClientError) -> str:
    raw = str(exc)
    lowered = raw.lower()
    if "401" in lowered or "403" in lowered or "unauth" in lowered or "permission" in lowered:
        return "Language model authentication failed."
    if "404" in lowered or "not_found" in lowered or "no longer available" in lowered:
        return "The configured Gemini model was not found."
    return "The language model request was rejected."
