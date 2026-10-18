import json
import logging
from typing import Annotated

import httpx
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from ragas.exceptions import RagasException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.agents.workflow import AgentWorkflow
from app.api.schemas import EvaluationResponse, HealthResponse, IngestResponse, QueryRequest
from app.core.config import get_settings
from app.core.database import get_engine
from app.core.ollama import OllamaError
from app.eval.runner import run_evaluation
from app.rag.ingestion import ingest_document

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        async with get_engine().connect() as connection:
            await connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        logger.exception("Database health check failed")
        raise HTTPException(status_code=503, detail="Database is unavailable") from exc

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{get_settings().ollama_base_url.rstrip('/')}/api/tags"
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="Ollama is unavailable") from exc
    return HealthResponse(status="ok", database="ok", ollama="ok")


@router.post("/ingest", response_model=IngestResponse)
async def ingest(files: Annotated[list[UploadFile], File()]) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one file")
    settings = get_settings()
    results = []
    for file in files:
        filename = file.filename or ""
        if not filename.lower().endswith((".pdf", ".md", ".markdown")):
            raise HTTPException(status_code=415, detail=f"Unsupported file type: {filename}")
        content = await file.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"{filename} exceeds the {settings.max_upload_bytes}-byte upload limit",
            )
        try:
            results.append(await ingest_document(filename, content))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except OllamaError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            logger.exception("Failed to ingest %s", filename)
            raise HTTPException(status_code=500, detail="Document indexing failed") from exc
        finally:
            await file.close()
    return IngestResponse(files=results)


@router.post("/query/stream")
async def query_stream(request: QueryRequest, http_request: Request) -> StreamingResponse:
    async def event_stream():
        async for event in AgentWorkflow().stream(request.query):
            if await http_request.is_disconnected():
                logger.info("Client disconnected from agent stream")
                break
            yield (
                f"event: {event['event']}\n"
                f"data: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/eval/run", response_model=EvaluationResponse)
async def evaluate() -> EvaluationResponse:
    try:
        result = await run_evaluation()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OllamaError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (RagasException, SQLAlchemyError) as exc:
        logger.exception("Ragas evaluation failed")
        raise HTTPException(status_code=500, detail="Evaluation failed; see API logs") from exc
    return EvaluationResponse(**result)
