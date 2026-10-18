import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from app.core.config import get_settings


class OllamaError(RuntimeError):
    """Raised when Ollama returns an error or an unexpected response."""


def _url(path: str) -> str:
    return f"{get_settings().ollama_base_url.rstrip('/')}{path}"


async def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
            response = await client.post(
                _url("/api/embed"),
                json={"model": settings.ollama_embedding_model, "input": list(texts)},
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(f"Could not create embeddings with Ollama: {exc}") from exc

    payload = response.json()
    vectors = payload.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise OllamaError("Ollama returned an unexpected embedding response")
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != settings.embedding_dimensions:
            raise OllamaError(
                f"Embedding dimension mismatch; expected {settings.embedding_dimensions}"
            )
    return vectors


async def chat(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": get_settings().ollama_chat_model,
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
            response = await client.post(_url("/api/chat"), json=payload)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(f"Could not get a chat response from Ollama: {exc}") from exc
    data = response.json()
    message = data.get("message")
    if not isinstance(message, dict):
        raise OllamaError("Ollama returned an unexpected chat response")
    return message


async def stream_chat(messages: list[dict[str, Any]]) -> AsyncIterator[str]:
    payload = {
        "model": get_settings().ollama_chat_model,
        "messages": messages,
        "stream": True,
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
            async with client.stream("POST", _url("/api/chat"), json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    message = data.get("message", {})
                    content = message.get("content", "")
                    if content:
                        yield content
                    if data.get("error"):
                        raise OllamaError(f"Ollama stream failed: {data['error']}")
    except httpx.HTTPError as exc:
        raise OllamaError(f"Could not stream a chat response from Ollama: {exc}") from exc
