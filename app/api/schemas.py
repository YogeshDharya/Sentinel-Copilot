from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)


class IngestedFile(BaseModel):
    filename: str
    chunks_indexed: int
    source_hash: str


class IngestResponse(BaseModel):
    files: list[IngestedFile]


class HealthResponse(BaseModel):
    status: str
    database: str
    ollama: str


class EvaluationResponse(BaseModel):
    examples: int
    faithfulness: float
    context_recall: float
    answer_relevancy: float
