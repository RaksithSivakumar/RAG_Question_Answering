"""Tests for PDF and TXT document ingestion."""

from pathlib import Path

import fitz
import pytest

from app.loaders import DocumentLoadError, extract_text_from_bytes, extract_text_from_path, validate_filename

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "documents" / "sample_leave_policy.txt"


def test_validate_filename_accepts_pdf_and_txt() -> None:
    assert validate_filename("policy.pdf").suffix == ".pdf"
    assert validate_filename("notes.txt").suffix == ".txt"


def test_validate_filename_rejects_unsupported_extension() -> None:
    with pytest.raises(DocumentLoadError, match="Unsupported file type"):
        validate_filename("image.png")


def test_txt_loading_from_path() -> None:
    text = extract_text_from_path(SAMPLE_PATH)
    assert "20 days of annual leave" in text
    assert "3 working days" in text
    assert "Tuesday and Thursday" in text


def test_txt_loading_from_bytes() -> None:
    content = SAMPLE_PATH.read_bytes()
    text = extract_text_from_bytes("sample_leave_policy.txt", content)
    assert "Remote work is allowed" in text


def test_empty_txt_is_rejected() -> None:
    with pytest.raises(DocumentLoadError, match="empty"):
        extract_text_from_bytes("empty.txt", b"   \n\n  ")


def test_empty_file_bytes_are_rejected() -> None:
    with pytest.raises(DocumentLoadError, match="empty"):
        extract_text_from_bytes("empty.txt", b"")


def test_pdf_text_extraction() -> None:
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Employees receive 20 days of annual leave per calendar year.")
    content = pdf.tobytes()
    pdf.close()

    text = extract_text_from_bytes("policy.pdf", content)
    assert "20 days of annual leave" in text


def test_empty_pdf_is_rejected() -> None:
    pdf = fitz.open()
    pdf.new_page()
    content = pdf.tobytes()
    pdf.close()

    with pytest.raises(DocumentLoadError, match="empty"):
        extract_text_from_bytes("blank.pdf", content)
