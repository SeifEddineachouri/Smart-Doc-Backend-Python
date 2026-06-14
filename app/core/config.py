import os
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_env_file() -> str:
    # Allow overriding .env location while keeping a stable default from any working directory.
    explicit_env_file = os.getenv("SMARTDOC_AI_ENV_FILE")
    if explicit_env_file:
        return explicit_env_file
    return str(Path(__file__).resolve().parents[2] / ".env")


class Settings(BaseSettings):
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"),
    )
    service_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SERVICE_TOKEN", "APP_AI_SERVICE_TOKEN", "APP_AI_GATEWAY_SERVICE_TOKEN"),
    )
    model_name: str = Field(default="gemini-2.5-flash", validation_alias=AliasChoices("MODEL_NAME", "GEMINI_MODEL_NAME"))
    embedding_model_name: str = Field(
        default="gemini-embedding-001",
        validation_alias=AliasChoices("EMBEDDING_MODEL_NAME", "GEMINI_EMBEDDING_MODEL_NAME"),
    )
    embedding_dimensions: int = Field(default=768, validation_alias=AliasChoices("EMBEDDING_DIMENSIONS"))
    embedding_query_task_type: str = "RETRIEVAL_QUERY"
    embedding_document_task_type: str = "RETRIEVAL_DOCUMENT"
    chunk_size_words: int = 220
    chunk_overlap_words: int = 40
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.20
    retrieval_lexical_weight: float = 0.15
    retrieval_log_scores: bool = True

    model_config = SettingsConfigDict(env_file=_resolve_env_file(), env_file_encoding="utf-8", extra="ignore")

    @property
    def env_file_path(self) -> str:
        return _resolve_env_file()


settings = Settings()

