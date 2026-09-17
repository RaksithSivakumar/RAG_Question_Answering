"""Document loaders for PDF and TXT files."""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from app.config import ALLOWED_EXTENSIONS

SUPPORTED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "application/octet-stream",
}


class DocumentLoadError(ValueError):
    """Raised when a document cannot be accepted or parsed."""


def validate_filename(filename: str | None) -> Path:
    if not filename or not filename.strip():
        raise DocumentLoadError("A filename is required.")

    path = Path(filename)
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise DocumentLoadError(
            f"Unsupported file type '{suffix or 'unknown'}'. Allowed types: {allowed}."
        )
    return path


def extract_text_from_bytes(filename: str, content: bytes) -> str:
    """Extract text from an uploaded file's raw bytes."""
    path = validate_filename(filename)
    suffix = path.suffix.lower()

    if not content:
        raise DocumentLoadError("The uploaded file is empty.")

    if suffix == ".txt":
        text = _decode_txt(content)
    elif suffix == ".pdf":
        text = _extract_pdf(content)
    else:
        raise DocumentLoadError(f"Unsupported file type '{suffix}'.")

    cleaned = _normalize_text(text)
    if not cleaned:
        raise DocumentLoadError("Extracted text is empty.")
    return cleaned


def extract_text_from_path(file_path: str | Path) -> str:
    path = Path(file_path)
    validate_filename(path.name)
    if not path.is_file():
        raise DocumentLoadError(f"File not found: {path}")
    return extract_text_from_bytes(path.name, path.read_bytes())


def _decode_txt(content: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentLoadError("Unable to decode text file.")


def _extract_pdf(content: bytes) -> str:
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises various errors for corrupt files
        raise DocumentLoadError("Unable to read PDF file.") from exc

    try:
        pages = [page.get_text("text") for page in document]
    finally:
        document.close()
    return "\n".join(pages)


def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    collapsed = "\n".join(line for line in lines if line)
    return collapsed.strip()
