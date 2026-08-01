"""Configuration loaded independently of the caller's working directory."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIMENSIONS: dict[str, int] = {"jina-embeddings-v3": 1024}


class ConfigurationError(ValueError):
    """A human-readable configuration error that callers can safely display."""


class Settings(BaseSettings):
    """Application settings. Optional integrations are validated only when used."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    google_api_key: SecretStr | None = None
    jina_api_key: SecretStr | None = None
    postgres_host: str | None = None
    postgres_port: int = 5432
    postgres_db: str | None = None
    postgres_user: str | None = None
    postgres_password: SecretStr | None = None
    embedding_provider: Literal["jina", "gemini"] = "jina"
    embedding_model: str = "jina-embeddings-v3"
    llm_model: str = "gemini-2.5-flash"
    chunk_size: int = Field(default=600, ge=100)
    chunk_overlap: int = Field(default=120, ge=0)
    metadata_generation_enabled: bool = True
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    request_timeout_seconds: float = Field(default=60, gt=0)
    max_retries: int = Field(default=3, ge=1, le=10)

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_smaller_than_size(cls, value: int, info: object) -> int:
        size = getattr(info, "data", {}).get("chunk_size")
        if size is not None and value >= size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return value

    @property
    def embedding_dimensions(self) -> int:
        """Return the model's fixed vector size; it is never user-configurable."""
        try:
            return MODEL_DIMENSIONS[self.embedding_model]
        except KeyError as error:
            supported = ", ".join(sorted(MODEL_DIMENSIONS))
            raise ConfigurationError(
                f"Unsupported embedding model '{self.embedding_model}'. "
                f"Supported models: {supported}."
            ) from error

    def validate_ai_requirements(self) -> None:
        """Validate credentials only for the integrations enabled in this command."""
        if self.embedding_provider == "jina" and not self.jina_api_key:
            raise ConfigurationError("JINA_API_KEY is required for EMBEDDING_PROVIDER=jina. Add it to .env.")
        if self.embedding_provider == "gemini":
            raise ConfigurationError(
                "Gemini embeddings are not configured in MODEL_DIMENSIONS. Use "
                "EMBEDDING_PROVIDER=jina with jina-embeddings-v3."
            )
        if self.metadata_generation_enabled and not self.google_api_key:
            raise ConfigurationError(
                "GOOGLE_API_KEY is required when METADATA_GENERATION_ENABLED=true. Add it to .env."
            )
        _ = self.embedding_dimensions

    def validate_database_requirements(self) -> None:
        """Ensure all PostgreSQL settings exist before a database connection is created."""
        missing = [name for name, value in {
            "POSTGRES_HOST": self.postgres_host, "POSTGRES_DB": self.postgres_db,
            "POSTGRES_USER": self.postgres_user, "POSTGRES_PASSWORD": self.postgres_password,
        }.items() if not value]
        if missing:
            raise ConfigurationError(
                "Database configuration is required for full ingestion. Missing: " + ", ".join(missing)
            )

    @cached_property
    def database_url(self) -> URL:
        self.validate_database_requirements()
        assert self.postgres_host and self.postgres_db and self.postgres_user and self.postgres_password
        return URL.create(
            "postgresql+psycopg", username=self.postgres_user,
            password=self.postgres_password.get_secret_value(), host=self.postgres_host,
            port=self.postgres_port, database=self.postgres_db,
        )
