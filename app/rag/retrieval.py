from typing import TypedDict

from sqlalchemy import text

from app.core.config import get_settings
from app.core.database import get_engine
from app.core.ollama import embed_texts


class SearchHit(TypedDict):
    source_name: str
    content: str
    score: float


async def search_knowledge_base(query: str, top_k: int | None = None) -> list[SearchHit]:
    settings = get_settings()
    limit = top_k or settings.retrieval_top_k
    if limit < 1 or limit > 20:
        raise ValueError("top_k must be between 1 and 20")
    vectors = await embed_texts([query])
    vector = "[" + ",".join(str(value) for value in vectors[0]) + "]"
    statement = text(
        """
        WITH vector_ranked AS (
            SELECT id, source_name, content,
                   row_number() OVER (ORDER BY embedding <=> CAST(:embedding AS vector)) AS rank
            FROM knowledge_chunks
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :candidate_limit
        ),
        text_ranked AS (
            SELECT id, source_name, content,
                   row_number() OVER (
                       ORDER BY ts_rank_cd(to_tsvector('english', content),
                                           websearch_to_tsquery('english', :query)) DESC
                   ) AS rank
            FROM knowledge_chunks
            WHERE to_tsvector('english', content) @@ websearch_to_tsquery('english', :query)
            ORDER BY ts_rank_cd(to_tsvector('english', content),
                                websearch_to_tsquery('english', :query)) DESC
            LIMIT :candidate_limit
        ),
        fused AS (
            SELECT id, source_name, content, 1.0 / (60 + rank) AS score FROM vector_ranked
            UNION ALL
            SELECT id, source_name, content, 1.0 / (60 + rank) AS score FROM text_ranked
        )
        SELECT source_name, content, SUM(score) AS score
        FROM fused
        GROUP BY id, source_name, content
        ORDER BY score DESC
        LIMIT :top_k
        """
    )
    async with get_engine().connect() as connection:
        result = await connection.execute(
            statement,
            {
                "embedding": vector,
                "query": query,
                "candidate_limit": max(limit * 4, 20),
                "top_k": limit,
            },
        )
        return [
            {"source_name": row.source_name, "content": row.content, "score": float(row.score)}
            for row in result
        ]
