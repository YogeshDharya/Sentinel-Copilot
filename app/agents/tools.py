import json
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import text

from app.core.database import get_engine
from app.rag.retrieval import search_knowledge_base


class SearchKnowledgeInput(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class DatabaseStatsInput(BaseModel):
    metric: Literal["document_count", "chunk_count", "recent_sources"]


class FinalReportInput(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=4000)
    findings: list[str] = Field(default_factory=list, max_length=20)
    sources: list[str] = Field(default_factory=list, max_length=20)


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search indexed internal documents for evidence relevant to the user's request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Focused knowledge-base search query",
                    },
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_database_stats",
            "description": "Read safe aggregate statistics about the indexed knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["document_count", "chunk_count", "recent_sources"],
                    }
                },
                "required": ["metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "format_final_report",
            "description": (
                "Format verified findings and source names as a concise Markdown report."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "findings": {"type": "array", "items": {"type": "string"}},
                    "sources": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "summary"],
            },
        },
    },
]


async def query_database_stats(raw_arguments: dict[str, Any]) -> str:
    arguments = DatabaseStatsInput.model_validate(raw_arguments)
    queries = {
        "document_count": """
            SELECT count(DISTINCT source_hash) AS value FROM knowledge_chunks
        """,
        "chunk_count": "SELECT count(*) AS value FROM knowledge_chunks",
        "recent_sources": """
            SELECT source_name, max(created_at) AS indexed_at
            FROM knowledge_chunks GROUP BY source_name ORDER BY indexed_at DESC LIMIT 10
        """,
    }
    async with get_engine().connect() as connection:
        result = await connection.execute(text(queries[arguments.metric]))
        if arguments.metric == "recent_sources":
            payload = [
                {"source_name": row.source_name, "indexed_at": row.indexed_at.isoformat()}
                for row in result
            ]
        else:
            payload = {"metric": arguments.metric, "value": result.scalar_one()}
    return json.dumps(payload)


async def dispatch_tool(name: str, raw_arguments: dict[str, Any]) -> str:
    if name == "search_knowledge_base":
        arguments = SearchKnowledgeInput.model_validate(raw_arguments)
        hits = await search_knowledge_base(arguments.query, arguments.top_k)
        return json.dumps(hits, ensure_ascii=False)
    if name == "query_database_stats":
        return await query_database_stats(raw_arguments)
    if name == "format_final_report":
        arguments = FinalReportInput.model_validate(raw_arguments)
        lines = [f"# {arguments.title}", "", arguments.summary]
        if arguments.findings:
            lines.extend(["", "## Findings", *[f"- {item}" for item in arguments.findings]])
        if arguments.sources:
            lines.extend(["", "## Sources", *[f"- {source}" for source in arguments.sources]])
        return "\n".join(lines)
    raise ValueError(f"Unsupported tool requested: {name}")
