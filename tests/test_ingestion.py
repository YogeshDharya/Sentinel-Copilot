import pytest

from app.rag.ingestion import extract_text


def test_extracts_markdown() -> None:
    assert extract_text("notes.md", b"# Title\n\nContent") == "# Title\n\nContent"


def test_rejects_unsupported_file_type() -> None:
    with pytest.raises(ValueError, match="Only PDF and Markdown"):
        extract_text("notes.txt", b"hello")


def test_rejects_invalid_markdown_encoding() -> None:
    with pytest.raises(ValueError, match="valid UTF-8"):
        extract_text("notes.md", b"\xff")
