import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.config import get_settings

logger = logging.getLogger(__name__)
_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    return _engine


async def initialize_database() -> None:
    dimensions = get_settings().embedding_dimensions
    async with get_engine().begin() as connection:
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await connection.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    source_name TEXT NOT NULL,
                    source_hash CHAR(64) NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({dimensions}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (source_hash, chunk_index)
                )
                """
            )
        )
        await connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_embedding "
                "ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)"
            )
        )
        await connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_search "
                "ON knowledge_chunks USING gin (to_tsvector('english', content))"
            )
        )
    logger.info("Database schema is ready")


async def close_database() -> None:
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
