import pytest

from app.rag.chunking import chunk_text


def test_chunk_text_covers_input_and_preserves_overlap() -> None:
    text = "one two three four five six seven eight nine"

    chunks = chunk_text(text, chunk_size=18, overlap=5)

    assert len(chunks) > 1
    assert "".join(chunks).replace(" ", "") != ""
    assert all(len(chunk) <= 18 for chunk in chunks)
    assert chunks[0].split()[-1] in chunks[1].split()[:2]


def test_chunk_text_returns_no_chunks_for_blank_input() -> None:
    assert chunk_text(" \n ", chunk_size=10, overlap=2) == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (10, -1), (10, 10)],
)
def test_chunk_text_rejects_invalid_sizes(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size, overlap)
