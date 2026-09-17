"""Recursive character-based text chunking."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import CHUNK_OVERLAP, CHUNK_SIZE

# Split from coarsest to finest so chunks prefer paragraph/sentence boundaries.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    text: str
    position: int
    document_name: str


def chunk_text(
    text: str,
    document_name: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    cleaned = text.strip()
    if not cleaned:
        return []

    units = _recursive_split(cleaned, chunk_size, SEPARATORS)
    merged = _merge_with_overlap(units, chunk_size, chunk_overlap)

    chunks: list[TextChunk] = []
    for index, piece in enumerate(merged):
        chunks.append(
            TextChunk(
                chunk_id=f"{document_name}::chunk-{index}",
                text=piece,
                position=index,
                document_name=document_name,
            )
        )
    return chunks


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Split oversized text on the next separator; do not pack to chunk_size yet."""
    separator = separators[0]
    remaining = separators[1:]

    if separator == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size) if text[i : i + chunk_size]]

    parts = text.split(separator)
    units: list[str] = []
    for part in parts:
        if not part:
            continue
        if len(part) <= chunk_size:
            units.append(part)
        elif remaining:
            units.extend(_recursive_split(part, chunk_size, remaining))
        else:
            units.extend(_recursive_split(part, chunk_size, [""]))
    return units


def _join(parts: list[str], separator: str = " ") -> str:
    return separator.join(part for part in parts if part)


def _merge_with_overlap(units: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    if not units:
        return []

    separator = " "
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def total_with(piece: str) -> int:
        extra = len(separator) if current else 0
        return current_len + extra + len(piece)

    for unit in units:
        if current and total_with(unit) > chunk_size:
            chunks.append(_join(current, separator))
            while current and current_len > chunk_overlap:
                dropped = current.pop(0)
                current_len -= len(dropped)
                if current:
                    current_len -= len(separator)
            if current and total_with(unit) > chunk_size:
                current = []
                current_len = 0
        extra = len(separator) if current else 0
        current.append(unit)
        current_len += extra + len(unit)

    if current:
        chunks.append(_join(current, separator))
    return chunks
