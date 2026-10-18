from functools import lru_cache
from pathlib import Path

from pydantic import Field, PositiveInt, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+asyncpg://sentinel:sentinel@localhost:5432/sentinel"
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.2"
    ollama_embedding_model: str = "nomic-embed-text"
    embedding_dimensions: PositiveInt = 768
    chunk_size: PositiveInt = 1000
    chunk_overlap: int = Field(default=150, ge=0)
    retrieval_top_k: PositiveInt = 5
    max_upload_bytes: PositiveInt = 20 * 1024 * 1024
    max_agent_tool_rounds: PositiveInt = 5
    evaluation_dataset_path: Path = Path("app/eval/examples.json")
    log_level: str = "INFO"

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
