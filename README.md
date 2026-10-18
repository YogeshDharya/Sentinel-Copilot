# Sentinel Copilot

A local-first enterprise knowledge and agentic copilot: FastAPI, PostgreSQL + pgvector, Ollama, and Ragas.

## Project layout

```text
Sentinel-Copilot/
├── app/
│   ├── agents/       # Tool-calling workflow and guarded tools
│   ├── api/          # HTTP endpoints and request/response schemas
│   ├── core/         # Settings, database, and Ollama client
│   ├── eval/         # Ragas evaluation runner and example dataset
│   ├── rag/          # Parsing, chunking, indexing, and hybrid retrieval
│   └── main.py       # FastAPI application and lifecycle
├── examples/         # Sample source document to ingest
├── tests/            # Unit tests for deterministic core behavior
├── .env.example
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

## Run locally with Docker

Requirements: Docker Desktop with Compose and Python 3.11+ for local development.

1. From this directory, copy `.env.example` to `.env` and adjust values if needed.
2. Start the stack:

   ```powershell
   docker compose up --build -d
   ```

3. Pull the chat and embedding models into Ollama (first run downloads several GB):

   ```powershell
   docker compose exec ollama ollama pull llama3.2
   docker compose exec ollama ollama pull nomic-embed-text
   ```

4. Open `http://localhost:8000/docs`. The API waits for PostgreSQL but Ollama models
   must be pulled before using ingestion, querying, or evaluation.
5. Upload `examples/employee_handbook.md` using `POST /ingest`, then try
   `POST /query/stream`. `GET /eval/run` runs the included Ragas examples against
   the indexed sample handbook.

The API container initializes the pgvector extension and database schema on startup.
PostgreSQL and the API are bound to localhost by default. Ollama is available at
`http://localhost:11434` for local tooling.

## Run without Docker (Python API)

Start PostgreSQL with pgvector and Ollama separately, pull the same models, then set
`DATABASE_URL` and `OLLAMA_BASE_URL` from `.env.example`. From this directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Run unit tests with `pytest`.

## API

- `GET /health` — liveness and dependency readiness.
- `POST /ingest` — multipart upload of PDF or Markdown files.
- `POST /query/stream` — JSON `{ "query": "..." }`; emits SSE events named
  `status`, `token`, `complete`, or `error`. Status events report tool activity,
  not hidden chain-of-thought.
- `GET /eval/run` — evaluates the included question/ground-truth examples using
  Faithfulness, Context Recall, and Answer Relevancy. Requires indexed relevant
  content and a working Ollama model.

The `query_database_stats` tool exposes only a small set of fixed, read-only
aggregate queries; it never accepts arbitrary SQL. Uploaded files are processed in
memory and are not persisted to disk. Search uses reciprocal-rank fusion (RRF) over
pgvector cosine similarity and PostgreSQL English full-text ranking.

## Operational notes

This is a local portfolio application, not a hardened multi-tenant deployment.
Before exposing it beyond localhost, add authentication/authorization, request
quotas, TLS, tenant isolation, audit retention, and a secrets manager. The default
embedding model returns 768 dimensions; changing the model may require changing
`EMBEDDING_DIMENSIONS` and rebuilding the database schema. The included sample
evaluation set is a starting point, not a substitute for a domain-specific,
curated benchmark.
