"""Chunk domain types."""

from typing import Any
from pydantic import BaseModel, Field


class SemanticChunk(BaseModel):
    """A contiguous, timestamped semantic transcript chunk."""

    chunk_index: int = Field(ge=0)
    start_second: int = Field(ge=0)
    end_second: int = Field(ge=0)
    chunk_text: str = Field(min_length=1)


class PersistedChunk(SemanticChunk):
    """Chunk enriched with generated metadata and its vector."""

    metadata: dict[str, Any] | None
    embedding: list[float]
