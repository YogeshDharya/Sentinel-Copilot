import hashlib
import io
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sqlalchemy import text

from app.core.config import get_settings
from app.core.database import get_engine
from app.core.ollama import embed_texts
from app.rag.chunking import chunk_text


@dataclass(frozen=True)
class IngestionResult:
    filename: str
    chunks_indexed: int
    source_hash: str


def extract_text(filename: str, content: bytes) -> str:
    suffix = filename.rsplit(".", maxsplit=1)[-1].lower() if "." in filename else ""
    if suffix in {"md", "markdown"}:
        try:
            return content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{filename} is not valid UTF-8 Markdown") from exc
    if suffix == "pdf":
        try:
            reader = PdfReader(io.BytesIO(content))
            pages = [page.extract_text() or "" for page in reader.pages]
        except (PdfReadError, OSError, ValueError) as exc:
            raise ValueError(f"{filename} is not a readable PDF") from exc
        extracted = "\n\n".join(pages).strip()
        if not extracted:
            raise ValueError(f"{filename} has no extractable text (OCR is not enabled)")
        return extracted
    raise ValueError("Only PDF and Markdown files are supported")


async def ingest_document(filename: str, content: bytes) -> IngestionResult:
    settings = get_settings()
    source_hash = hashlib.sha256(content).hexdigest()
    pieces = chunk_text(
        extract_text(filename, content),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    if not pieces:
        raise ValueError(f"{filename} contains no indexable text")

    vectors: list[list[float]] = []
    for offset in range(0, len(pieces), 32):
        vectors.extend(await embed_texts(pieces[offset : offset + 32]))

    rows = [
        {
            "source_name": filename,
            "source_hash": source_hash,
            "chunk_index": index,
            "content": piece,
            "embedding": "[" + ",".join(str(value) for value in vector) + "]",
        }
        for index, (piece, vector) in enumerate(zip(pieces, vectors, strict=True))
    ]
    async with get_engine().begin() as connection:
        await connection.execute(
            text(
                """
                INSERT INTO knowledge_chunks
                    (source_name, source_hash, chunk_index, content, embedding)
                VALUES
                    (:source_name, :source_hash, :chunk_index, :content, CAST(:embedding AS vector))
                ON CONFLICT (source_hash, chunk_index) DO UPDATE SET
                    source_name = EXCLUDED.source_name,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    created_at = now()
                """
            ),
            rows,
        )
    return IngestionResult(filename=filename, chunks_indexed=len(rows), source_hash=source_hash)
